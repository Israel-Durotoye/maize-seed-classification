#!/usr/bin/env python3
"""Live maize seed sorting dashboard: camera -> model -> per-seed counts, served over HTTP/WebSocket.

Run from the project root:  .venv/bin/python web/backend/server.py --camera 0
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import csv
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
# Same preprocessing, camera reader and per-seed latch as the physical sorter.
from main import CLASSES, DecisionGate, LatestCamera, interpret_probabilities, prepare_rgb  # noqa: E402

LOG = logging.getLogger("maize.web")
MODEL_DIR = PROJECT_DIR / "model_outputs"
FRONTEND_DIST = PROJECT_DIR / "web" / "frontend" / "dist"
EVENT_LOG = PROJECT_DIR / "web_sort_events.jsonl"
RUN_DIR_PATTERN = re.compile(r"^\d{8}T\d{6}_\d+Z$")
MODEL_NAMES = ("maize_mobilenetv3large_final.keras", "best_finetuned.keras")

# Notebook defaults (section 15): used for the empty-view test and when no threshold CSV exists.
EMPTY_CONFIDENCE, EMPTY_MARGIN = 0.90, 0.30
FALLBACK_THRESHOLD = {"confidence": 0.90, "margin": 0.50}

# Gate timing, matching main.py.
STABLE_FRAMES, STABLE_SECONDS = 3, 0.15
EMPTY_FRAMES, EMPTY_SECONDS = 3, 0.4
COOLDOWN, MAX_FRAME_GAP = 1.0, 1.0
WIDTH, HEIGHT = 640, 480


def newest_model() -> tuple[Path, Path]:
    """Return (model file, its run directory), preferring the newest timestamped training run."""
    runs = sorted((p for p in MODEL_DIR.iterdir() if p.is_dir() and RUN_DIR_PATTERN.match(p.name)),
                  key=lambda p: p.name, reverse=True)
    for directory in [*runs, MODEL_DIR]:
        for name in MODEL_NAMES:
            if (directory / name).is_file():
                return directory / name, directory
    raise FileNotFoundError(f"No Keras model found in {MODEL_DIR}")


def load_policy(run_dir: Path) -> dict:
    """Recreate the notebook's validation-calibrated per-class thresholds from its search CSV."""
    thresholds = {}
    path = run_dir / "threshold_search_results.csv"
    rows = list(csv.DictReader(path.open())) if path.is_file() else []
    for name in CLASSES[:2]:
        candidates = [r for r in rows if r["class_name"] == name and int(r["accepted"]) >= 3
                      and float(r["precision"]) >= 0.95 and float(r["no_maize_false_acceptance"]) <= 0.005]
        candidates.sort(key=lambda r: (-int(r["correct"]), -float(r["precision"]),
                                       -float(r["confidence"]), -float(r["margin"])))
        best = candidates[0] if candidates else None
        thresholds[name] = ({"confidence": float(best["confidence"]), "margin": float(best["margin"])}
                            if best else dict(FALLBACK_THRESHOLD))
    return {"actuation_enabled": True, "class_thresholds": thresholds,
            "empty_confidence": EMPTY_CONFIDENCE, "empty_margin": EMPTY_MARGIN}


class KerasClassifier:
    def __init__(self, model_path: Path, policy: dict):
        import keras
        import tensorflow as tf
        self.model = keras.saving.load_model(model_path, compile=False)
        self.size = tuple(int(x) for x in self.model.input_shape[1:3])
        self.metadata = {"decision_policy": policy}
        self._infer = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)
        self._infer(np.zeros((1, *self.size, 3), np.float32))  # Warm-up trace.

    def predict(self, rgb: Image.Image):
        probabilities = self._infer(prepare_rgb(rgb, self.size)).numpy()
        return interpret_probabilities(probabilities, self.metadata)


def parse_source(value: str):
    return int(value) if value.isdigit() else value


class Engine:
    """Owns the camera and model in a background thread; everything else reads snapshots."""
    def __init__(self, classifier: KerasClassifier, model_info: dict, source, roi=None):
        self.classifier, self.model_info, self.roi = classifier, model_info, roi
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.source = source
        self.switch_to = None
        self.camera: LatestCamera | None = None
        self.counts = {"GOOD_SEED": 0, "BAD_SEED": 0, "uncertain": 0}
        self.events: deque[dict] = deque(maxlen=25)
        self.next_id = 1
        self.prediction = None
        self.status, self.message = "starting", "Opening camera…"
        self.armed = False
        self.fps = self.inference_ms = 0.0
        self.gate = self._new_gate()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _new_gate():
        return DecisionGate(STABLE_FRAMES, STABLE_SECONDS, EMPTY_FRAMES, EMPTY_SECONDS, COOLDOWN, MAX_FRAME_GAP)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.camera:
            self.camera.close()

    def reset(self):
        with self.lock:
            self.counts = {"GOOD_SEED": 0, "BAD_SEED": 0, "uncertain": 0}
            self.events.clear()
            self.next_id = 1

    def request_camera(self, source):
        with self.lock:
            self.switch_to = source

    def latest_frame(self):
        camera = self.camera
        if camera is None:
            return None
        try:
            frame, _ = camera.latest()
        except Exception:
            return None
        if frame is not None and self.roi:
            import cv2
            x, y, w, h = self.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (64, 200, 255), 2)
        return frame

    def snapshot(self) -> dict:
        with self.lock:
            p = self.prediction
            counts = dict(self.counts)
            return {
                "status": self.status, "message": self.message, "armed": self.armed,
                "camera": str(self.source), "fps": round(self.fps, 1), "inference_ms": round(self.inference_ms, 1),
                "counts": {"good": counts["GOOD_SEED"], "bad": counts["BAD_SEED"], "uncertain": counts["uncertain"],
                           "total": counts["GOOD_SEED"] + counts["BAD_SEED"]},
                "prediction": None if p is None else {
                    "label": p.label, "confidence": p.confidence, "margin": p.margin,
                    "probabilities": dict(zip(CLASSES, p.probabilities)), "eligible": p.eligible, "empty": p.empty},
                "events": list(self.events),
            }

    def _set_status(self, status, message):
        with self.lock:
            self.status, self.message = status, message

    def _open_camera(self):
        if self.camera:
            self.camera.close()
            self.camera = None
        self._set_status("starting", f"Opening camera {self.source}…")
        self.camera = LatestCamera(self.source, WIDTH, HEIGHT)
        self.gate = self._new_gate()

    def _record(self, label, reason, prediction):
        with self.lock:
            # A seed that left without a confident class goes to the sorter's reject bin, but it may
            # also be a hand or shadow, so the dashboard reports it separately from Bad seeds.
            if reason == "uncertain_reject":
                label = "UNSURE"
                self.counts["uncertain"] += 1
            else:
                self.counts[label] += 1
            event = {"id": self.next_id, "time": time.time(), "label": label, "reason": reason,
                     "confidence": prediction.confidence if prediction.label == label else None}
            self.next_id += 1
            self.events.appendleft(event)
        with EVENT_LOG.open("a") as f:
            f.write(json.dumps({**event, "camera": str(self.source), "model": self.model_info["path"]}) + "\n")
        LOG.info("Seed #%d counted as %s (%s)", event["id"], label, reason)

    def _run(self):
        last_used = 0.0
        frame_times: deque[float] = deque(maxlen=30)
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    switch, self.switch_to = self.switch_to, None
                if switch is not None:
                    self.source = switch
                    self._open_camera()
                elif self.camera is None:
                    self._open_camera()
                frame, captured = self.camera.latest()
                if frame is None or captured <= last_used:
                    time.sleep(0.005)
                    continue
                last_used = captured
                crop = frame
                if self.roi:
                    x, y, w, h = self.roi
                    crop = frame[y:y + h, x:x + w]
                rgb = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
                started = time.perf_counter()
                prediction = self.classifier.predict(rgb)
                elapsed = (time.perf_counter() - started) * 1000
                label = self.gate.observe(prediction, captured)
                if label:
                    self._record(label, self.gate.event_reason, prediction)
                frame_times.append(time.monotonic())
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
                with self.lock:
                    self.prediction, self.armed = prediction, self.gate.armed
                    self.inference_ms = elapsed if not self.inference_ms else 0.8 * self.inference_ms + 0.2 * elapsed
                    self.fps = fps
                    self.status = "armed" if self.gate.armed else "waiting"
                    self.message = ("Ready: place one seed in view" if self.gate.armed
                                    else "Clear the view: waiting for an empty tray between seeds")
            except Exception as exc:  # Camera unplugged, permission denied, etc. Keep retrying.
                LOG.warning("Camera/inference problem: %s", exc)
                self._set_status("error", str(exc))
                if self.camera:
                    self.camera.close()
                    self.camera = None
                self.stop_event.wait(2.0)


def probe_cameras(limit=6, skip=None):
    import cv2
    found = []
    for index in range(limit):
        if index == skip:
            found.append(index)
            continue
        capture = cv2.VideoCapture(index)
        ok = capture.isOpened() and capture.read()[0]
        capture.release()
        if ok:
            found.append(index)
    return found


def create_app(engine: Engine):
    import cv2

    app = FastAPI(title="Maize Seed Sorter")

    @app.get("/api/state")
    def state():
        return engine.snapshot()

    @app.get("/api/model")
    def model():
        return engine.model_info

    @app.post("/api/reset")
    def reset():
        engine.reset()
        return engine.snapshot()

    @app.get("/api/cameras")
    def cameras():
        current = engine.source if isinstance(engine.source, int) else None
        return {"current": str(engine.source), "available": probe_cameras(skip=current)}

    @app.post("/api/camera")
    def camera(source: str = Body(..., embed=True)):
        if not str(source).strip():
            raise HTTPException(400, "Camera source is required.")
        engine.request_camera(parse_source(str(source).strip()))
        return {"ok": True}

    @app.get("/api/stream")
    def stream():
        def frames():
            blank = None
            while True:
                frame = engine.latest_frame()
                if frame is None:
                    if blank is None:
                        blank = cv2.imencode(".jpg", np.full((HEIGHT, WIDTH, 3), 24, np.uint8))[1].tobytes()
                    jpeg = blank
                    time.sleep(0.25)
                else:
                    jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    time.sleep(1 / 25)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                await socket.send_json(engine.snapshot())
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            pass

    if FRONTEND_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            target = FRONTEND_DIST / path
            if path and target.is_file() and FRONTEND_DIST in target.resolve().parents:
                return FileResponse(target)
            return FileResponse(FRONTEND_DIST / "index.html")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=os.environ.get("MAIZE_CAMERA", "0"),
                        help="Camera index (0 = first camera, 1 = second…) or a device path/URL.")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to view from other devices on the network.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Optional crop of the camera frame that the model looks at.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_path, run_dir = newest_model()
    policy = load_policy(run_dir)
    LOG.info("Loading %s", model_path.relative_to(PROJECT_DIR))
    classifier = KerasClassifier(model_path, policy)
    metrics_path = run_dir / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    model_info = {"path": str(model_path.relative_to(PROJECT_DIR)), "run": run_dir.name,
                  "architecture": "MobileNetV3Large", "input_size": list(classifier.size),
                  "classes": CLASSES, "policy": policy,
                  "test_accuracy": metrics.get("accuracy"), "test_macro_f1": metrics.get("macro_f1")}
    LOG.info("Thresholds: %s", json.dumps(policy["class_thresholds"]))

    engine = Engine(classifier, model_info, parse_source(args.camera), args.roi)
    engine.start()
    if not FRONTEND_DIST.is_dir():
        LOG.warning("Frontend not built: run `npm install && npm run build` in web/frontend.")
    import uvicorn
    LOG.info("Dashboard: http://%s:%d", "127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port)
    try:
        uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
