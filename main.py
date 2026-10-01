#!/usr/bin/env python3
"""Pi camera classification, FIFO seed tracking, and ESP8266 feeder control."""
from __future__ import annotations

import binascii
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import secrets
import signal
import threading
import time

import numpy as np
from PIL import Image, ImageOps

# EDIT THESE SETTINGS, THEN RUN: python main.py
PROJECT_DIR = Path(__file__).resolve().parent

# Everyday operation
DRY_RUN = True                  # True = camera/classification only; False = control the ESP.
SERIAL_PORT = "/dev/ttyUSB0"      # Check the Pi's port; /dev/serial/by-id/... is more stable.
CAMERA = 0                      # USB camera index, or a path such as "/dev/video0".
HEADLESS = True                  # True for SSH; False to show the camera window.
FEED = False                    # True starts both motors after a stable empty camera view.
CAMERA_TO_SERVO_MM = 500.0       # 50 cm, measured from the camera's seed detection line.
SERVO_LEAD_SECONDS = 2.0         # Command before arrival; includes inference/serial/movement time.
SEED_CLEARANCE_MM = 10.0         # Distance a seed must pass beyond the servo before switching.
MAX_QUEUED_SEEDS = 64
BELT_SPEED_MM_S = 10.0
SEED_SPACING_MM = 70.0

# Optional tests: choose only one; leave all three as None for normal camera use.
IMAGE_PATH = None                # Example: PROJECT_DIR / "example_seed.jpg"
SERVO_TEST = None                # "GOOD" or "BAD"; set DRY_RUN = False to move the servo.
MOTOR_TEST = None                # "CONVEYOR", "DISK" or "BOTH"; set DRY_RUN = False to move.
TEST_SECONDS = 5.0               # Motor test duration, up to 30 seconds.

# Model and camera settings
MODEL_PATH = PROJECT_DIR / "model_outputs/maize_mobilenetv3large_float32.tflite"
METADATA_PATH = PROJECT_DIR / "model_outputs/deployment_metadata.json"
EVENT_LOG = PROJECT_DIR / "sort_events.jsonl"
BAUD = 115200
WIDTH = 640
HEIGHT = 480
ROI = None                      # Optional crop: (x, y, width, height), e.g. (160, 80, 320, 320).
THREADS = 2

# Decision timing (seconds unless the name says FRAMES)
STABLE_FRAMES = 3
STABLE_SECONDS = 0.15
EMPTY_FRAMES = 3
EMPTY_SECONDS = 0.4
COOLDOWN = 1.0
MAX_FRAME_AGE = 0.75
MAX_FRAME_GAP = 1.0

LOG = logging.getLogger("maize")
CLASSES = ["BAD_SEED", "GOOD_SEED", "NO_MAIZE"]


@dataclass(frozen=True)
class FeedPlan:
    """Direct-drive 40 mm roller, 200-step motors at 1/32, six-hole disk."""
    conveyor_period_us: int
    disk_period_us: int

    @classmethod
    def calculate(cls, speed_mm_s: float, spacing_mm: float):
        if not all(math.isfinite(v) and v > 0 for v in (speed_mm_s, spacing_mm)):
            raise ValueError("Belt speed and seed spacing must be finite and positive.")
        belt_hz = speed_mm_s * 6400 / (math.pi * 40)
        disk_hz = speed_mm_s / spacing_mm * 6400 / 6
        if not all(math.isfinite(hz) and 10 <= hz <= 1000 for hz in (belt_hz, disk_hz)):
            raise ValueError("Both motors must stay within 10–1000 pulses/s; adjust speed/spacing.")
        periods = tuple(round(1_000_000 / hz) for hz in (belt_hz, disk_hz))
        return cls(*periods)

    @property
    def belt_speed_mm_s(self):
        return 1_000_000 / self.conveyor_period_us * math.pi * 40 / 6400

    @property
    def seed_interval_s(self):
        return self.disk_period_us / 1_000_000 * 6400 / 6

    def validate_timing(self, settle_ms):
        # Travel time can span many seed intervals: the FIFO tracks all of them.
        clearance = SEED_CLEARANCE_MM / self.belt_speed_mm_s
        detection = max(COOLDOWN, EMPTY_SECONDS) + STABLE_SECONDS + 2 * MAX_FRAME_AGE
        required = max(SERVO_LEAD_SECONDS + clearance, detection)
        if self.seed_interval_s <= required:
            raise ValueError(f"Feed interval {self.seed_interval_s:.2f}s must exceed "
                             f"the {required:.2f}s detection/servo clearance budget.")
        if SERVO_LEAD_SECONDS <= settle_ms / 1000 + MAX_FRAME_AGE + 0.1:
            raise ValueError("SERVO_LEAD_SECONDS must cover servo settling, inference and serial margin.")


def encode_message(payload: str) -> bytes:
    raw = payload.encode("ascii")
    return raw + f"|{binascii.crc_hqx(raw, 0xFFFF):04X}\n".encode("ascii")


def decode_message(line: bytes) -> list[str] | None:
    try:
        raw, checksum = line.rstrip(b"\r\n").rsplit(b"|", 1)
        if len(checksum) != 4 or binascii.crc_hqx(raw, 0xFFFF) != int(checksum, 16):
            return None
        return raw.decode("ascii").split()
    except (ValueError, UnicodeError):
        return None


def prepare_rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Exact notebook contract: EXIF-correct RGB, Pillow bilinear, float32 0..255."""
    height, width = size
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(rgb.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)[None]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float
    probabilities: tuple[float, ...]
    eligible: bool
    empty: bool


def interpret_probabilities(values, metadata: dict) -> Prediction:
    probs = np.asarray(values, dtype=np.float64).reshape(-1)
    if probs.shape != (3,) or not np.isfinite(probs).all():
        raise ValueError("Model must return three finite class probabilities.")
    if np.any(probs < 0) or np.any(probs > 1) or abs(float(probs.sum()) - 1) > 0.01:
        raise ValueError("Model output is not a three-class softmax probability vector.")
    index = int(np.argmax(probs))
    confidence = float(probs[index])
    margin = confidence - float(np.sort(probs)[-2])
    label = CLASSES[index]
    policy = metadata["decision_policy"]
    threshold = policy["class_thresholds"].get(label, {})
    eligible = label != "NO_MAIZE" and bool(policy["actuation_enabled"]) and (
        confidence >= threshold.get("confidence", 1.0)
        and margin >= threshold.get("margin", 1.0)
    )
    empty = label == "NO_MAIZE" and confidence >= policy["empty_confidence"] and margin >= policy["empty_margin"]
    return Prediction(label, confidence, margin, tuple(float(x) for x in probs), eligible, empty)


def validate_metadata(metadata: dict) -> None:
    if metadata.get("schema_version") != 2:
        raise ValueError("Use schema-version-2 deployment_metadata.json from the updated notebook.")
    if metadata.get("class_names") != CLASSES or metadata.get("class_to_index") != dict(zip(CLASSES, range(3))):
        raise ValueError("Model class mapping does not match BAD_SEED, GOOD_SEED, NO_MAIZE.")
    if (metadata.get("input_color_order") != "RGB" or metadata.get("input_dtype") != "float32"
            or metadata.get("external_input_range") != [0.0, 255.0]
            or metadata.get("resize_method") != "pillow_bilinear"):
        raise ValueError("Unsupported model preprocessing contract.")
    size = metadata.get("image_size", [])
    if len(size) != 2 or any(type(x) is not int or x < 1 for x in size):
        raise ValueError("Invalid image_size in metadata.")
    policy = metadata["decision_policy"]
    if type(policy["actuation_enabled"]) is not bool:
        raise ValueError("actuation_enabled must be a boolean.")
    numbers = [policy["empty_confidence"], policy["empty_margin"]]
    for name in CLASSES[:2]:
        numbers.extend(policy["class_thresholds"][name][key] for key in ("confidence", "margin"))
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or not 0 <= x <= 1 for x in numbers):
        raise ValueError("All decision thresholds must be finite numbers between 0 and 1.")


class Classifier:
    def __init__(self, model_path: Path, metadata_path: Path, threads: int = 2):
        self.metadata = json.loads(metadata_path.read_text())
        validate_metadata(self.metadata)
        model_bytes = model_path.read_bytes()
        expected = self.metadata.get("tflite_sha256", {}).get(model_path.name)
        if not expected or hashlib.sha256(model_bytes).hexdigest() != expected:
            raise ValueError("Model checksum mismatch. Copy the model and metadata from the SAME Colab run; keep filenames.")
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                try:
                    from tensorflow.lite import Interpreter
                except ImportError as exc:
                    raise RuntimeError("Install ai-edge-litert, tflite-runtime, or TensorFlow for your Python/CPU platform.") from exc
        self.interpreter = Interpreter(model_content=model_bytes, num_threads=threads)
        self.interpreter.allocate_tensors()
        inputs, outputs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one model input and one output.")
        self.input, self.output = inputs[0], outputs[0]
        self.size = tuple(self.metadata["image_size"])
        if list(self.input["shape"]) != [1, *self.size, 3] or list(self.output["shape"]) != [1, 3]:
            raise ValueError("Unexpected model tensor shape.")
        if self.input["dtype"] != np.float32 or self.output["dtype"] != np.float32:
            raise ValueError("Use the float32 or dynamic-range export with float32 input/output.")

    def predict(self, rgb: Image.Image) -> Prediction:
        self.interpreter.set_tensor(self.input["index"], prepare_rgb(rgb, self.size))
        self.interpreter.invoke()
        return interpret_probabilities(self.interpreter.get_tensor(self.output["index"]), self.metadata)


class DecisionGate:
    """Latch one action per observed seed; start disarmed until a stable empty view."""
    def __init__(self, stable_frames=3, stable_seconds=0.15, empty_frames=3,
                 empty_seconds=0.4, cooldown=1.0, max_gap=1.0):
        self.stable_frames, self.stable_seconds = stable_frames, stable_seconds
        self.empty_frames, self.empty_seconds = empty_frames, empty_seconds
        self.cooldown, self.max_gap = cooldown, max_gap
        self.armed = False
        self.last_action = float("-inf")
        self.last_frame = None
        self.candidate = None
        self.count = self.empty_count = 0
        self.since = self.empty_since = 0.0
        self.seed_seen_at = None
        self.event_seen_at = None
        self.event_reason = None

    def reset_streak(self):
        self.candidate, self.count, self.empty_count = None, 0, 0

    def emit(self, label, now, reason):
        self.event_seen_at = self.seed_seen_at
        self.event_reason = reason
        self.seed_seen_at = None
        self.armed = False
        self.last_action = now
        self.reset_streak()
        return label

    def reject_pending(self, now):
        if self.seed_seen_at is not None:
            return self.emit("BAD_SEED", now, "uncertain_reject")
        return None

    def observe(self, prediction: Prediction, now: float) -> str | None:
        if self.last_frame is not None and (now <= self.last_frame or now - self.last_frame > self.max_gap):
            self.reset_streak()
            self.armed = False
            self.seed_seen_at = None
        self.last_frame = now
        if prediction.empty:
            self.candidate, self.count = None, 0
            if not self.empty_count:
                self.empty_since = now
            self.empty_count += 1
            if (self.seed_seen_at is not None and self.empty_count >= self.empty_frames
                    and now - self.empty_since >= self.empty_seconds):
                return self.reject_pending(now)
            if (self.empty_count >= self.empty_frames and now - self.empty_since >= self.empty_seconds
                    and now - self.last_action >= self.cooldown):
                self.armed = True
            return None
        self.empty_count = 0
        if self.armed and self.seed_seen_at is None:
            self.seed_seen_at = now  # First non-empty frame, not the later confirming frame.
        if not self.armed or not prediction.eligible:
            self.candidate, self.count = None, 0
            return None
        if prediction.label != self.candidate:
            self.candidate, self.count, self.since = prediction.label, 1, now
        else:
            self.count += 1
        if self.count >= self.stable_frames and now - self.since >= self.stable_seconds:
            return self.emit(prediction.label, now, "classified")
        return None


@dataclass
class TrackedSeed:
    seed_id: int
    label: str
    seen_at: float
    arrival_at: float
    command_at: float
    clear_at: float
    sequence: int | None = None
    ready: bool = False


class SeedFIFO:
    """Constant-speed position estimate; stop on lost timing instead of draining overdue seeds."""
    def __init__(self, speed_mm_s, settle_ms=350):
        self.travel_seconds = CAMERA_TO_SERVO_MM / speed_mm_s
        self.clearance_seconds = SEED_CLEARANCE_MM / speed_mm_s
        self.settle_seconds = settle_ms / 1000
        if self.travel_seconds <= SERVO_LEAD_SECONDS + MAX_FRAME_AGE:
            raise ValueError("Camera travel time must exceed servo lead plus inference time.")
        if SERVO_LEAD_SECONDS <= self.settle_seconds + MAX_FRAME_AGE + 0.1:
            raise ValueError("SERVO_LEAD_SECONDS is too short for inference and servo movement.")
        self.queue = deque()
        self.next_id = 1
        self.last_arrival = float("-inf")

    def enqueue(self, label, seen_at):
        if label not in CLASSES[:2] or seen_at is None or not math.isfinite(seen_at):
            raise ValueError("A queued seed needs GOOD/BAD and a finite detection timestamp.")
        if len(self.queue) >= MAX_QUEUED_SEEDS:
            raise RuntimeError("Seed FIFO is full; stop and clear the belt before restarting.")
        arrival = seen_at + self.travel_seconds
        due = arrival - SERVO_LEAD_SECONDS
        if due <= time.monotonic():
            raise TimeoutError("Seed was classified too late for its sorting deadline.")
        if due <= self.last_arrival + self.clearance_seconds:
            raise RuntimeError("Seeds are too close or out of order for servo switching; stop feeding.")
        seed = TrackedSeed(self.next_id, label, seen_at, arrival, due,
                           arrival + self.clearance_seconds)
        self.next_id += 1
        self.last_arrival = arrival
        self.queue.append(seed)
        return seed

    def service(self, link):
        if link:
            link.poll()  # Also advances a pending sort without blocking camera inference.
        now = time.monotonic()
        events = []
        while self.queue:
            seed = self.queue[0]
            if seed.sequence is not None and link and link.completed_sequence == seed.sequence:
                seed.ready = True
            if not seed.ready and now >= seed.arrival_at:
                raise TimeoutError("Seed reached its estimated arrival before confirmed servo readiness.")
            if seed.ready and now >= seed.clear_at:
                self.queue.popleft()
                events.append(("seed_passed_estimate", seed))
                continue
            if seed.sequence is None and not seed.ready and now >= seed.command_at:
                if now + self.settle_seconds + 0.1 >= seed.arrival_at:
                    raise TimeoutError("Missed FIFO servo deadline; stop and clear the belt.")
                if link:
                    seed.sequence = link.begin_sort(seed.label, deadline=seed.arrival_at - 0.1)
                else:
                    seed.ready = True
                events.append(("servo_command" if link else "dry_run_command", seed))
            break
        return events


class SerialLink:
    """One pending servo command, polled without blocking inference; retries reuse IDs."""
    def __init__(self, port: str, baud: int = 115200, ack_timeout: float = 0.4, retries: int = 2):
        import serial
        self.serial = serial.Serial(port, baudrate=baud, timeout=0, write_timeout=0.5, exclusive=True)
        self.session = secrets.token_hex(8).upper()
        self.boot = None
        self.capabilities = set()
        self.sequence = 0
        self.completed_sequence = 0
        self.pending_sort = None
        self.buffer = bytearray()
        self.discarding = False
        self.ack_timeout, self.retries = ack_timeout, retries
        self.last_ping = self.last_rx = time.monotonic()
        try:
            # USB-UART boards often reset on open. HELLO itself cannot sort a seed.
            deadline, next_hello = time.monotonic() + 8.0, 0.0
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_hello:
                    self.send(f"HELLO {self.session}")
                    next_hello = now + 0.5
                for fields in self.receive():
                    if 5 <= len(fields) <= 7 and fields[:2] == ["READY", self.session]:
                        if len(fields[2]) != 8 or any(c not in "0123456789ABCDEF" for c in fields[2]):
                            raise RuntimeError("Invalid ESP boot identifier.")
                        self.boot = fields[2]
                        self.capabilities = set(fields[5:])
                        if "HOLD_V1" not in self.capabilities or fields[3] != "0":
                            raise RuntimeError("Upload the new hold-position ESP sketch; the old neutral-return firmware cannot run FIFO sorting.")
                        self.settle_ms = int(fields[4])
                        if not 50 <= self.settle_ms <= 3000:
                            raise RuntimeError("ESP reported invalid servo timing.")
                        self.last_rx = time.monotonic()
                        LOG.info("ESP ready: boot=%s holds position; settle=%dms", self.boot, self.settle_ms)
                        return
                time.sleep(0.01)
            raise TimeoutError("ESP handshake timed out. Check port, cable, baud and uploaded sketch.")
        except Exception:
            self.serial.close()
            raise

    def send(self, payload):
        frame = encode_message(payload)
        if self.serial.write(frame) != len(frame):
            raise IOError("Incomplete serial write; delivery is uncertain.")

    def receive(self):
        messages = []
        for byte in self.serial.read(min(self.serial.in_waiting, 4096)):
            if byte == 10:
                if not self.discarding:
                    fields = decode_message(bytes(self.buffer))
                    if fields:
                        messages.append(fields)
                self.buffer.clear()
                self.discarding = False
            elif not self.discarding:
                if len(self.buffer) >= 192:
                    self.buffer.clear()
                    self.discarding = True
                else:
                    self.buffer.append(byte)
        return messages

    def poll(self):
        now = time.monotonic()
        if now - self.last_ping >= 0.5:
            self.send(f"PING {self.session} {self.boot}")
            self.last_ping = now
        valid = []
        for fields in self.receive():
            if fields[0] in {"BOOT", "FAULT"}:
                raise RuntimeError(f"ESP reset/fault: {' '.join(fields)}. Sorting stopped; restart after inspection.")
            if fields[0] == "ERR":
                raise RuntimeError(f"ESP rejected command: {' '.join(fields)}")
            if len(fields) >= 3 and fields[1:3] == [self.session, self.boot]:
                self.last_rx = now
                valid.append(fields)
        if now - self.last_rx > 2.0:
            raise TimeoutError("ESP heartbeat lost. Sorting stopped; no automatic reconnect or replay.")
        self.advance_sort(valid, now)
        return valid

    def begin_sort(self, label: str, deadline=None):
        if label not in CLASSES[:2]:
            raise ValueError("Only GOOD_SEED and BAD_SEED can actuate.")
        if self.pending_sort is not None:
            raise RuntimeError("A servo command is already pending.")
        now = time.monotonic()
        duration = self.settle_ms / 1000
        if deadline is not None and now + duration >= deadline:
            raise TimeoutError("Insufficient time to position the servo before seed arrival.")
        self.sequence += 1
        if self.sequence > 0xFFFFFFFF:
            raise RuntimeError("Serial sequence exhausted; restart with an empty view.")
        command = f"SORT {self.session} {self.boot} {self.sequence} {'GOOD' if label == 'GOOD_SEED' else 'BAD'}"
        self.send(command)
        timeout = now + duration + (self.retries + 2) * self.ack_timeout + 1.0
        self.pending_sort = dict(command=command, started=now, last_send=now, attempts=0,
                                 accepted=False, deadline=min(timeout, deadline) if deadline is not None else timeout)
        return self.sequence

    def advance_sort(self, messages, now):
        pending = self.pending_sort
        if pending is None:
            return
        if now >= pending["deadline"]:
            raise TimeoutError("Servo completion deadline missed; stop and clear the belt.")
        for fields in messages:
            if len(fields) == 5 and fields[0] == "ACK" and fields[3] == str(self.sequence):
                if fields[4] == "DONE":
                    self.completed_sequence = self.sequence
                    self.pending_sort = None
                    return
                if fields[4] == "ACCEPTED":
                    pending["accepted"] = True
        duration = self.settle_ms / 1000
        retry_due = (now - pending["last_send"] >= self.ack_timeout
                     and (not pending["accepted"] or now - pending["started"] >= duration + self.ack_timeout))
        if retry_due and pending["attempts"] < self.retries:
            if now + duration >= pending["deadline"]:
                raise TimeoutError("No time for a servo retry before seed arrival.")
            self.send(pending["command"])
            pending["attempts"] += 1
            pending["last_send"] = now

    def sort(self, label: str):
        """Blocking convenience for the manual servo test only."""
        self.begin_sort(label)
        while self.pending_sort is not None:
            self.poll()
            time.sleep(0.005)

    def start_feed(self, plan: FeedPlan, motor="BOTH"):
        if "FEED_V1" not in self.capabilities:
            raise RuntimeError("Upload the integrated ESP8266 firmware before using motor control.")
        if motor not in {"BOTH", "CONVEYOR", "DISK"}:
            raise ValueError("Unknown motor selection.")
        belt = plan.conveyor_period_us if motor != "DISK" else 0
        disk = plan.disk_period_us if motor != "CONVEYOR" else 0
        command = f"RUN {self.session} {self.boot} {belt} {disk}"
        expected = ["RUNNING", self.session, self.boot, str(belt), str(disk)]
        for _ in range(self.retries + 1):
            self.send(command)  # Same parameters: ESP acknowledges without restarting motors.
            deadline = time.monotonic() + self.ack_timeout
            while time.monotonic() < deadline:
                if expected in self.poll():
                    return
                time.sleep(0.005)
        raise TimeoutError("Motor startup acknowledgement missing; stop and inspect before retrying.")

    def close(self):
        try:
            if self.boot:
                self.send(f"HOME {self.session} {self.boot}")
                deadline = time.monotonic() + self.ack_timeout
                while time.monotonic() < deadline:
                    if ["HOMED"] in self.receive():
                        break
                    time.sleep(0.005)
                else:
                    LOG.warning("HOME acknowledgement missing; ESP watchdog/lease will stop feeding.")
        except Exception:
            LOG.exception("Could not send HOME; ESP watchdog will stop feeding and request neutral.")
        finally:
            self.serial.close()


class LatestCamera:
    """Drain camera continuously so inference and servo waits do not build a frame queue."""
    def __init__(self, source, width, height):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open camera {source!r}.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.timestamp = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        failures = 0
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError("Camera stopped delivering frames.")
                    self.stop.wait(0.05)
                    continue
                failures = 0
                with self.lock:
                    self.frame, self.timestamp = frame, time.monotonic()
        except Exception as exc:
            with self.lock:
                self.error = exc

    def latest(self):
        with self.lock:
            if self.error:
                raise self.error
            return (None if self.frame is None else self.frame.copy()), self.timestamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)
        if not self.thread.is_alive():
            self.capture.release()
        # A stuck backend read is left to process teardown; never race release() against read().


def validate_settings():
    """Catch configuration mistakes before opening the camera or moving hardware."""
    for name in ("DRY_RUN", "HEADLESS", "FEED"):
        if type(globals()[name]) is not bool:
            raise ValueError(f"{name} must be True or False, without quotes.")
    if SERVO_TEST not in (None, "GOOD", "BAD"):
        raise ValueError('SERVO_TEST must be None, "GOOD" or "BAD".')
    if MOTOR_TEST not in (None, "CONVEYOR", "DISK", "BOTH"):
        raise ValueError('MOTOR_TEST must be None, "CONVEYOR", "DISK" or "BOTH".')
    if sum(v is not None for v in (IMAGE_PATH, SERVO_TEST, MOTOR_TEST)) > 1:
        raise ValueError("Choose only one of IMAGE_PATH, SERVO_TEST and MOTOR_TEST.")
    if IMAGE_PATH is not None and not str(IMAGE_PATH).strip():
        raise ValueError("IMAGE_PATH must be an image filename or None.")
    if FEED and any(v is not None for v in (IMAGE_PATH, SERVO_TEST, MOTOR_TEST)):
        raise ValueError("FEED is only for normal camera operation; turn it off for tests.")
    if not DRY_RUN and IMAGE_PATH is None and not SERIAL_PORT:
        raise ValueError("Set SERIAL_PORT to the ESP's device path.")
    for name in ("WIDTH", "HEIGHT", "THREADS", "STABLE_FRAMES", "EMPTY_FRAMES", "BAUD", "MAX_QUEUED_SEEDS"):
        value = globals()[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("STABLE_SECONDS", "EMPTY_SECONDS", "COOLDOWN", "MAX_FRAME_AGE", "MAX_FRAME_GAP",
                 "CAMERA_TO_SERVO_MM", "SERVO_LEAD_SECONDS", "SEED_CLEARANCE_MM", "BELT_SPEED_MM_S", "SEED_SPACING_MM"):
        value = globals()[name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(TEST_SECONDS) or not 0 < TEST_SECONDS <= 30:
        raise ValueError("TEST_SECONDS must be greater than 0 and no more than 30.")
    if ROI is not None:
        if (not isinstance(ROI, (tuple, list)) or len(ROI) != 4
                or any(type(v) is not int for v in ROI)
                or min(ROI[:2]) < 0 or min(ROI[2:]) <= 0):
            raise ValueError("ROI must be None or (x, y, width, height) with nonnegative origin and positive size.")
    if not ((type(CAMERA) is int and CAMERA >= 0) or (isinstance(CAMERA, str) and CAMERA.strip())):
        raise ValueError('CAMERA must be a nonnegative camera index or a device path such as "/dev/video0".')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    def stop_requested(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_requested)
    link = camera = event_file = fifo = None
    try:
        validate_settings()
        feed_plan = FeedPlan.calculate(BELT_SPEED_MM_S, SEED_SPACING_MM) if FEED or MOTOR_TEST else None
        if MOTOR_TEST:
            LOG.info("Motor test %s for %.1fs%s", MOTOR_TEST, TEST_SECONDS,
                     " (DRY RUN)" if DRY_RUN else "")
            if not DRY_RUN:
                link = SerialLink(SERIAL_PORT, BAUD)
                link.start_feed(feed_plan, MOTOR_TEST)
                deadline = time.monotonic() + TEST_SECONDS
                while time.monotonic() < deadline:
                    link.poll()
                    time.sleep(0.01)
            return 0
        if SERVO_TEST:
            LOG.info("Manual %s servo position for %.1fs%s", SERVO_TEST, TEST_SECONDS, " (DRY RUN)" if DRY_RUN else "")
            if not DRY_RUN:
                link = SerialLink(SERIAL_PORT, BAUD)
                link.sort(SERVO_TEST + "_SEED")
                deadline = time.monotonic() + TEST_SECONDS
                while time.monotonic() < deadline:
                    link.poll()
                    time.sleep(0.01)
            return 0
        classifier = Classifier(Path(MODEL_PATH), Path(METADATA_PATH), THREADS)
        if IMAGE_PATH:
            with Image.open(IMAGE_PATH) as image:
                prediction = classifier.predict(image)
            print(json.dumps(prediction.__dict__, indent=2))
            return 0
        if not DRY_RUN and not classifier.metadata["decision_policy"]["actuation_enabled"]:
            raise RuntimeError("Validation did not produce an acceptable sorting policy. Set DRY_RUN = True and improve/retrain the dataset.")
        import cv2
        event_file = Path(EVENT_LOG).open("a", buffering=1)
        if not DRY_RUN:
            link = SerialLink(SERIAL_PORT, BAUD)
        if FEED:
            if link and "FEED_V1" not in link.capabilities:
                raise RuntimeError("Upload the integrated ESP8266 firmware before setting FEED = True.")
            feed_plan.validate_timing(link.settle_ms if link else 350)
            LOG.info("Feed configured: belt %.3f mm/s, seed every %.3fs; waiting for empty view.",
                     feed_plan.belt_speed_mm_s, feed_plan.seed_interval_s)
        fifo = SeedFIFO(feed_plan.belt_speed_mm_s if FEED else BELT_SPEED_MM_S,
                        link.settle_ms if link else 350)
        LOG.info("FIFO travel: %.1f mm / %.3f mm/s = %.3fs. Start with the entire belt empty.",
                 CAMERA_TO_SERVO_MM, feed_plan.belt_speed_mm_s if FEED else BELT_SPEED_MM_S, fifo.travel_seconds)

        def service_fifo():
            for event, seed in fifo.service(link):
                event_file.write(json.dumps({"time": time.time(), "event": event, "dry_run": DRY_RUN,
                                            "seed_id": seed.seed_id, "class": seed.label,
                                            "sequence": seed.sequence, "arrival_at": seed.arrival_at}) + "\n")
                LOG.info("Seed %d: %s %s | FIFO=%d", seed.seed_id, seed.label, event, len(fifo.queue))

        camera = LatestCamera(CAMERA, WIDTH, HEIGHT)
        gate = DecisionGate(STABLE_FRAMES, STABLE_SECONDS, EMPTY_FRAMES,
                            EMPTY_SECONDS, COOLDOWN, MAX_FRAME_GAP)
        LOG.info("%s. Show an EMPTY view first; leave an empty gap between seeds.", "DRY RUN" if DRY_RUN else "Sorter connected")
        last_used = 0.0
        last_new = time.monotonic()
        last_status = 0.0
        feed_started = False
        while True:
            service_fifo()
            frame, captured = camera.latest()
            now = time.monotonic()
            if frame is None or captured <= last_used:
                if now - last_new > MAX_FRAME_GAP:
                    raise TimeoutError("Camera frame gap exceeded; FIFO tracking stopped.")
                time.sleep(0.005)
                continue
            if last_used and captured - last_used > MAX_FRAME_GAP:
                raise TimeoutError("Camera frame gap exceeded; seed identity may be lost.")
            last_used, last_new = captured, now
            if now - captured > MAX_FRAME_AGE:
                raise TimeoutError("Camera frame is stale; stopping sorting.")
            crop = frame
            if ROI:
                x, y, width, height = ROI
                if x + width > frame.shape[1] or y + height > frame.shape[0]:
                    raise ValueError("ROI extends outside the actual camera frame.")
                crop = frame[y:y + height, x:x + width]
            rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            prediction = classifier.predict(rgb)
            if time.monotonic() - captured > MAX_FRAME_AGE:
                raise TimeoutError("Inference is too slow for the configured frame-age limit; no command sent.")
            action = gate.observe(prediction, captured)
            # A seed that never becomes confident must not inherit the previous GOOD outlet.
            if (action is None and gate.seed_seen_at is not None
                    and time.monotonic() >= gate.seed_seen_at + fifo.travel_seconds
                    - SERVO_LEAD_SECONDS - 2 * MAX_FRAME_AGE):
                action = gate.reject_pending(captured)
            if FEED and not feed_started and gate.armed:
                if link:
                    link.start_feed(feed_plan)
                feed_started = True
                LOG.info("Feeding started%s", " (DRY RUN)" if DRY_RUN else "")
            if action:
                seed = fifo.enqueue(action, gate.event_seen_at)
                record = {"time": time.time(), "event": "seed_queued", "dry_run": DRY_RUN,
                          "seed_id": seed.seed_id, "class": action, "reason": gate.event_reason,
                          "seen_at": seed.seen_at, "arrival_at": seed.arrival_at, "command_at": seed.command_at,
                          "prediction": prediction.__dict__, "model": str(MODEL_PATH)}
                event_file.write(json.dumps(record) + "\n")
                LOG.info("Queued seed %d: %s (%s), arrival in %.2fs | FIFO=%d", seed.seed_id,
                         action, gate.event_reason, seed.arrival_at - time.monotonic(), len(fifo.queue))
            service_fifo()
            if now - last_status >= 1.0:
                LOG.info("%s %.3f | %s", prediction.label, prediction.confidence, "ARMED" if gate.armed else "WAITING FOR EMPTY VIEW")
                last_status = now
            if not HEADLESS:
                display = frame.copy()
                if ROI:
                    x, y, width, height = ROI
                    cv2.rectangle(display, (x, y), (x + width, y + height), (255, 255, 0), 2)
                text = f"{prediction.label} {prediction.confidence:.2f} | FIFO {len(fifo.queue)} | {'ARMED' if gate.armed else 'WAIT EMPTY'}"
                cv2.putText(display, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Single-seed maize sorter (q to quit)", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        return 0
    except KeyboardInterrupt:
        LOG.info("Stopped by user.")
        return 0
    except Exception:
        LOG.exception("Sorting stopped.")
        return 1
    finally:
        if link:
            link.close()
        if fifo and fifo.queue:
            LOG.warning("Discarding %d tracked seeds. Clear the entire belt before restarting.", len(fifo.queue))
            fifo.queue.clear()
        if camera:
            camera.close()
            if not HEADLESS:
                camera.cv2.destroyAllWindows()
        if event_file:
            event_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
