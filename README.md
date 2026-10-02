# FIFO maize sorter

This project trains a three-class MobileNetV3Large classifier in Colab and runs
it on a Raspberry Pi. The Pi classifies one seed at a time and sends checked,
acknowledged USB serial commands to an ESP8266. One integrated ESP sketch drives
the conveyor, six-hole feeding disk, sorting servo and inspection LED. The Pi
explicitly starts feeding after observing an empty view. Each detected seed enters
a FIFO that tracks its 500 mm journey to the servo while classification continues.
The servo holds its last GOOD/BAD position and only moves for a different class.
ESP32 builds retain servo-only support; the motor pinout is for ESP8266.

`BAD_SEED = 0`, `GOOD_SEED = 1`, `NO_MAIZE = 2` everywhere. **Confidence is not a
measured accuracy guarantee.** A camera can classify visible appearance; it
cannot establish germination viability or detect hidden defects from an ordinary
surface image.

## Files

| File | Purpose |
| --- | --- |
| `Maize_Seed_Classification_Model.ipynb` | Colab training, evaluation, threshold selection and verified export |
| `main.py` | Camera inference, FIFO seed tracking and ESP USB serial client |
| `esp32_maize_sorter/esp32_maize_sorter.ino` | Integrated ESP8266 firmware; ESP32 servo-only compatibility (folder name retained) |
| `maize_seed_classification_realtime.py` | Compatibility entry point that runs `main.py` |
| `requirements.txt` | Common Python dependencies; install one inference runtime separately |
| `tests/` | Inference, dataset, decision gate, serial fault and native firmware simulation tests |

## 1. Train in Colab

Open the updated notebook, select a GPU runtime, and run from the top. Drive is
expected to contain:

```text
MyDrive/Maize_seed_dataset/GOOD_SEED.zip
MyDrive/Maize_seed_dataset/BAD_SEED.zip
MyDrive/Maize_seed_dataset/NO_MAIZE.zip
```

The extracted class directories must be immediately under `MyDrive/MaizeData/`.
If the ZIP contains another enclosing directory, adjust that layout before
discovery. Existing extracted files are preserved. HEIC/HEIF originals are
converted before discovery; keep the real class labels correct.

Before training, inspect the example images. Use real good and bad maize on the
same range of backgrounds, and include empty views, stones, other grains and
other confusing objects in `NO_MAIZE`. Seed size within the frame and image
cropping should match the Pi camera view. The augmentation changes orientation
and lighting; it does not create independent seeds or new real backgrounds.

For photos of the same physical seed, burst, or recording session, provide
`MyDrive/MaizeData/source_groups.csv`:

```csv
relative_path,group_id
GOOD_SEED/photo_001.jpg,session_01
GOOD_SEED/photo_002.jpg,session_01
BAD_SEED/photo_003.jpg,session_02
```

Include every retained original in this CSV. Keep all related images under one
group ID, including images across classes when the group is a recording session.
There must be enough independent groups to represent all classes in all three
splits. The grouped split approximates 70/15/15; exact proportions depend on
group sizes. Without the CSV, the notebook stratifies individual originals and
cannot prevent leakage from different photos of the same seed.

The updated notebook:

- Fully decodes images, applies EXIF orientation, excludes unreadable files,
  removes exact decoded-pixel duplicates, and stops on conflicting duplicate labels.
- Converts all originals to lossless RGB images with the same Pillow bilinear
  resize used by the Pi. JPEG, PNG, WEBP and BMP inputs are handled consistently.
- Splits real originals first, then expands **each training class to 3,000**.
  Classes already above 3,000 keep their images. Evaluation images are not augmented.
- Stages images onto the Colab VM to avoid reading Google Drive every epoch.
- Uses pretrained MobileNetV3Large, a regularized head, clipped gradients, and
  low-rate fine-tuning with BatchNorm statistics frozen.
- Selects checkpoints using validation macro F1 so each class contributes
  equally, then chooses between the best head and fine-tuned checkpoints using
  validation macro F1 and validation loss. Test data never selects the checkpoint.
- Chooses separate good/bad confidence and margin thresholds on validation data.
  If either class cannot meet the empirical precision/false-acceptance criteria,
  the exported policy disables actuation. Dry-run classification remains available.
- Reports per-class test metrics, wrong good/bad actuations, non-maize false
  actuations, and a Wilson interval showing uncertainty from the small test set.
- Checks the float32 TFLite model against Keras on every test original. The
  optional dynamic-range model is bundled only if it passes the same checks.

Outputs go to a **new run directory**:

```text
MyDrive/MaizeData/model_outputs/<UTC-run-id>/
```

Inspect `test_metrics.json`, `test_predictions.csv`, and the confusion matrix.
The 95% validation precision and 1% non-maize false-acceptance targets are empirical
selection criteria, not statistically established field performance. In
particular, zero errors in a handful of non-maize test images is weak evidence.
Do not tune parameters repeatedly against the held-out test set.

Download **`maize_deployment.zip`** after the final export check. Extract it beside
`main.py` on the Pi:

```text
main.py
model_outputs/
    maize_mobilenetv3large_float32.tflite
    deployment_metadata.json
    test_metrics.json
```

Keep model filenames unchanged. The Pi checks their SHA-256 hashes against the
metadata. The old `Maize_classifier.tflite` in this repository is not the new
three-class deployment bundle and will be rejected.

## 2. Wire the ESP8266 mechanism

Use **one copy of this firmware** for all actuators. Do not upload a separate
motor program afterward: it would replace the sorter. Select the actual carrier
board in Arduino IDE; `ESP8266MOD` is the module marking, not the carrier name.
The Dx labels below use the NodeMCU/D1-mini mapping. For a NodeMCU carrier choose
**NodeMCU 1.0 (ESP-12E Module)** from the **esp8266** board package (build target:
core 3.1.2). Constants in the sketch use raw GPIO numbers.

| Function | Board label | GPIO | Connection |
| --- | --- | --- | --- |
| Conveyor / Motor 1 STEP | D5 | 14 | Motor driver STEP |
| Conveyor / Motor 1 DIR | D6 | 12 | Motor driver DIR |
| Disk / Motor 2 STEP | D1 | 5 | Motor driver STEP |
| Disk / Motor 2 DIR | D0 | 16 | Motor driver DIR |
| Positional servo signal | D2 | 4 | Servo signal |
| Inspection LED | D7 | 13 | Active-high, 3.3 V-compatible MOSFET gate circuit |

**D6 is assigned as Motor 1 DIR. D0 is assigned as Motor 2 DIR. Neither driver
DIR input should also be wired to 3.3 V.** Both directions default HIGH; change
`CONVEYOR_DIRECTION` or `DISK_DIRECTION` to LOW if a motor turns the wrong way.
GPIO16/D0 is only used as a static direction output.

Use STEP/DIR drivers compatible with 3.3 V logic and the configured **10 µs STEP
high pulse**. Set each driver's current limit for its motor and configure its
hardware microstep pins to **1/32** according to that driver's datasheet. The
firmware cannot select microstepping through STEP/DIR. Motor windings connect to
drivers, never directly to GPIO. Use a suitable motor supply and common signal
ground. Driver ENABLE/SLEEP/RESET wiring depends on the actual driver; no enable
pin is allocated here, so stopping pulses does not remove holding current.
Provide a physical motor-power stop. Fit suitable pulldowns on STEP inputs and
the MOSFET gate so they stay inactive while the ESP is resetting.

The LED is steady ON during a valid Pi session and OFF at boot, HOME or watchdog
shutdown. No LED dimming PWM is used. ESP8266's per-pin waveform generator gives
each stepper its own period and the servo a 20 ms period, avoiding shared
`analogWriteFreq` settings. Do not add libraries that take over timer1.

Use a **standard positional hobby servo**, such as the small SG90-style type,
not a continuous-rotation servo. With continuous rotation, pulse width controls
speed/direction instead of a known diverter position.

| Connection | Destination |
| --- | --- |
| Pi USB port | ESP development board USB connector, through a data cable |
| Servo signal (often orange/yellow) | Configured ESP GPIO; default **GPIO 4** |
| Servo positive (often red) | External regulated supply appropriate to the servo, typically 5 V |
| Servo ground (often brown/black) | External supply ground **and ESP GND** |

Size the external supply for the servo's stall current. Do not power the servo
from an ESP GPIO or the 3.3 V pin. The USB cable powers the ESP board; do not tie
the external servo supply's positive lead to the board's USB/5 V rail unless
your board's power design explicitly supports it. Verify wire colors and pulse
limits against the actual servo.

Default firmware settings:

| Setting | Default |
| --- | --- |
| Serial rate | 115200 baud |
| Signal pin | GPIO 4 |
| Good angle | 45 degrees |
| Bad angle | 135 degrees |
| Neutral angle | 90 degrees |
| Servo waveform | 50 Hz; ESP8266 uses explicit microsecond pulses, ESP32 uses 14-bit LEDC |
| Pulse range | 1000–2000 microseconds |
| Hold at sorting angle | Until a different class, explicit HOME, shutdown or fault |
| Settling time after an outlet change | 350 ms (calibrate against actual servo) |
| Host communication watchdog | 3000 ms |

Edit the constants at the top of the sketch. GPIO numbering and available pins
depend on the board. Select an exposed output-capable pin that is not used by
USB, flash, PSRAM or another peripheral. Swap `GOOD_ANGLE` and `BAD_ANGLE` if the
outlets are reversed. Start with the linkage unloaded and verify travel without
forcing the servo against a mechanical stop. The conservative default pulse
range may provide less travel than the printed angles suggest on some servos.

For this ESP8266 machine, install the **esp8266** board package, select the actual
board and port, then upload the `.ino` file. No external motor or servo library
is required. ESP32 Arduino core 2.x and 3.x branches remain available for a
servo-only setup with its GPIO pin checked against that ESP32 board.
For native-USB ESP32 variants, configure **USB CDC On Boot** when required by your
board so `Serial` is exposed through the connected USB port. Boards using a USB
UART bridge normally appear as `ttyUSB*`; native CDC boards often use `ttyACM*`.
Close Arduino Serial Monitor before starting the Pi program.

After installing the Pi dependencies, test each direction with the linkage
unloaded. These manual tests need no model or camera. The servo holds the selected
position for `TEST_SECONDS` (default 5), then the test exits and requests neutral.
During normal sorting there is no return to neutral between seeds:

Edit the settings at the top of `main.py`:

```python
DRY_RUN = False
SERIAL_PORT = "/dev/serial/by-id/YOUR_ESP_DEVICE"
FEED = False
SERVO_TEST = "GOOD"  # Then change to "BAD" and run again.
MOTOR_TEST = None
IMAGE_PATH = None
```

Run `python main.py` for each test. Set `SERVO_TEST = None` afterward.
`DRY_RUN = True` logs the requested direction without opening the serial port.

### Motor commissioning and mechanical calibration

Start with no seeds and test each mechanism separately. Edit these settings;
no model or camera is needed. Replace the port with the actual device:

```python
DRY_RUN = False
SERIAL_PORT = "/dev/serial/by-id/YOUR_ESP_DEVICE"
FEED = False
SERVO_TEST = None
IMAGE_PATH = None
MOTOR_TEST = "CONVEYOR"  # Run once, then try "DISK", then "BOTH".
TEST_SECONDS = 5.0
```

Run `python main.py` for each test. Set `MOTOR_TEST = None` afterward.

Each test stops the motors and switches off the LED on exit. Ctrl+C also requests
stop. Heartbeats keep a running test alive; losing the Pi stops feeding. Set
`DRY_RUN = True` to check settings without touching hardware. Test duration is capped
at 30 seconds. Do not load seeds until directions and servo travel are verified.

| Quantity | Value / assumption |
| --- | --- |
| Full steps per revolution | 200, both motors |
| Microstepping | 32, both drivers |
| Pulses per revolution | 6,400 |
| Conveyor roller diameter | 40 mm, direct drive, no slip |
| Conveyor calibration | 50.9296 pulses/mm |
| Disk | 180 mm diameter, six equally spaced holes, direct drive |
| Nominal seed spacing | 70 mm, one seed delivered by each hole |
| Initial belt speed | 10 mm/s |
| Conveyor pulse rate | About 509 pulses/s (4.77 rpm) |
| Disk pulse rate | About 152 pulses/s (1.43 rpm) |
| Seed interval | About 7 seconds |

The disk turns continuously; it does not round each 60° hole advance to an
integer pulse count. Periods are rounded to whole microseconds (approximately
70 mm spacing); actual seed delivery must be measured. Hole diameter, missed or
double-filled holes, gearing and belt slip can change delivery. The 180 mm disk
diameter and 1 m belt length do not determine seed spacing or camera travel time.
`BELT_SPEED_MM_S` and `SEED_SPACING_MM` set the nominal feed rates. Both
configured motor rates must be within 10–1000 pulses/s. There is no acceleration
ramp: these low startup rates are for commissioning; check for missed steps
under the real load before increasing speed. If the gearing or microstepping
changes, update `FeedPlan.calculate` in `main.py`.

The camera must show an empty gap between seeds. Set a narrow, fixed inspection
ROI so the first non-empty frame consistently means a seed crossed the same
physical line. `CAMERA_TO_SERVO_MM = 500.0` is measured from that detection line
to the servo. The FIFO uses the **first non-empty frame timestamp**, even if
confidence is confirmed several frames later. A broad view with variable entry
positions makes time-based arrival estimates unreliable.

At 10 mm/s, travel is approximately 50 seconds. `SERVO_LEAD_SECONDS = 2.0`
commands the outlet about two seconds before estimated arrival, allowing for
inference, serial delivery and the 350 ms servo settling time. Measure the
actual switching time; change `SETTLE_MS` in the sketch and the Pi's lead time
if necessary. `SEED_CLEARANCE_MM = 10.0` keeps the previous seed protected until
it has travelled 10 mm past the servo. This clearance must cover the actual seed
length and diverter geometry. Seed spacing must allow both clearance and the
next direction change.

## 3. Install and run on the Pi

Use 64-bit Raspberry Pi OS and Python 3.10 or newer. Create a fresh environment:

```bash
python3 -m venv .venv-pi
source .venv-pi/bin/activate
python -m pip install -r requirements.txt
python -m pip install ai-edge-litert
```

Install a runtime wheel compatible with the Pi's architecture and Python version.
If `ai-edge-litert` has no compatible wheel, use `tflite-runtime` or TensorFlow
for that environment. `main.py` tries those runtimes in that order. Do not install
all three unnecessarily. For a system OpenCV installation, use a venv with
`--system-site-packages` instead of installing a second OpenCV build. A CSI camera
must be exposed through an OpenCV-compatible capture backend; Picamera2-only
setups require a capture adapter and are not automatically handled here.

All configuration is in the labelled settings section at the top of `main.py`.
There are no command-line options. Start the program with:

```bash
python main.py
```

The default is `DRY_RUN = True`, `FEED = False`, `CAMERA = 0`, and
`HEADLESS = True`: camera classification without opening serial or moving
hardware. The trained deployment model and metadata must still be present.
Paths for the model, metadata and event log are anchored to the project folder,
so running the script from another directory still finds them.

Choose the operation by editing these settings, then run the same command:

| Operation | Settings in `main.py` |
| --- | --- |
| Still image | `IMAGE_PATH = PROJECT_DIR / "example_seed.jpg"` |
| Camera only | `IMAGE_PATH = None`, `SERVO_TEST = None`, `MOTOR_TEST = None`, `DRY_RUN = True`, `FEED = False` |
| Camera preview window | `HEADLESS = False` on the Pi desktop |
| Fixed inspection crop | `ROI = (160, 80, 320, 320)`; use `None` for the full frame |
| Live sorting on a belt running at the configured speed | `DRY_RUN = False`, `FEED = False`, correct `SERIAL_PORT`, all test settings `None` |
| Automatic feeding and sorting | As above, plus `FEED = True`; verify belt speed and the 500 mm detection-to-servo distance |

Still-image mode never opens serial or moves hardware. Only one of `IMAGE_PATH`,
`SERVO_TEST` and `MOTOR_TEST` may be set at a time. Set them all to `None` for
normal camera operation. An ROI crops one fixed area; there is no watershed or
multi-seed detection. Match the framing used for training.

Identify the actual ESP port:

```bash
ls -l /dev/serial/by-id/
```

Copy that persistent path into `SERIAL_PORT`. The default `/dev/ttyUSB0` is only
an example and may differ on your Pi. Keep `HEADLESS = True` over SSH. If the OS
denies serial access, grant your user the appropriate serial-device group
membership, typically `dialout`, and log in again. Avoid running the entire
sorter as root. `q` in the preview window, Ctrl+C, or SIGTERM requests shutdown,
stops the motors and sends a neutral servo command.

## 4. FIFO tracking and held servo position

All settings remain at the top of `main.py`; start with `python main.py`.
For automatic feeding after physical calibration:

```python
DRY_RUN = False
SERIAL_PORT = "/dev/serial/by-id/YOUR_ESP_DEVICE"
FEED = True
CAMERA_TO_SERVO_MM = 500.0
BELT_SPEED_MM_S = 10.0
SEED_SPACING_MM = 70.0
SERVO_LEAD_SECONDS = 2.0
SEED_CLEARANCE_MM = 10.0
MAX_QUEUED_SEEDS = 64
IMAGE_PATH = None
SERVO_TEST = None
MOTOR_TEST = None
```

**Upload the revised ESP sketch before running this Pi version.** The Pi checks
for `HOLD_V1` during the handshake and refuses the older firmware that returned
the servo to neutral after every seed.

1. Start with the **entire belt empty**, not just the camera view. Three confident
   `NO_MAIZE` frames spanning 0.4 seconds arm detection. With `FEED = True`, the
   Pi then starts both motors. `DRY_RUN = True` never starts hardware.
2. Each new seed is timestamped on its first non-empty frame. Three eligible
   predictions of the same class over at least 0.15 seconds confirm its label.
   This creates one FIFO entry with a unique ID, label, arrival and command time.
   A stable empty view and cooldown re-arm detection for the next seed.
3. Arrival is `first_seen + distance / belt_speed`. The command is scheduled at
   `arrival - SERVO_LEAD_SECONDS`. At the initial settings this is about **50 s
   travel**, with the command issued at about **48 s**. Several seeds may be in
   transit simultaneously; long travel no longer prevents feeding.
4. Camera inference continues throughout travel and servo movement. The oldest
   FIFO entry is serviced first. Serial acknowledgements and retries are polled
   without waiting through an entire movement in the camera loop.
5. The ESP holds **45° for GOOD** or **135° for BAD**. A new seed with the same
   class gets an acknowledgement without another PWM position update. A
   different class changes the outlet, waits the configured settling time and
   acknowledges readiness. There is no periodic return to 90°.
6. The head remains tracked until its estimated arrival plus clearance time.
   The next seed cannot trigger a switch before the previous seed clears.
   Sequence GOOD → GOOD → BAD therefore causes two outlet movements, with GOOD
   held continuously through the first two seeds.

An uncertain non-empty seed is queued to the **BAD/reject outlet** if it leaves
without a stable classification, or before its command deadline approaches.
This prevents an uncertain seed from inheriting a previous GOOD position. Such
entries are logged as `uncertain_reject`, not as confident bad classifications.
No extra event is created for repeated frames of that same seed.

The FIFO is bounded. Overflow, late classification, missed servo readiness,
seeds too close for switching, camera gaps, serial faults or ESP reset stop the
run and discard queued estimates. Do not resume a stopped queue: clear the
entire belt before restarting. HOME, shutdown and faults stop feeding and return
the servo to neutral; normal sorting holds its last outlet indefinitely.

Tracking estimates position from **constant belt speed**; there is no encoder
or seed-arrival sensor. Integrated feeding uses the speed calculated from the
actual rounded step period. With `FEED = False`, tracking assumes an externally
running belt at `BELT_SPEED_MM_S`. Stopping/slipping the belt, missed motor steps,
a seed rolling, or seeds overtaking breaks that estimate. Measure travel over
500 mm and verify timing with real seeds before live sorting. Add an encoder or
arrival sensor if speed/position cannot be kept predictable. Two touching seeds
or an incorrectly detected empty gap cannot be reliably distinguished by this
single-seed camera classifier alone.

## 5. Serial reliability and fault behavior

Messages are ASCII with a CRC16-CCITT-FALSE checksum and newline termination:

```text
PAYLOAD|CCCC\n
HELLO <host-session>
READY <host-session> <ESP-boot-id> 0 <settle-ms> HOLD_V1 FEED_V1
RUN <host-session> <ESP-boot-id> <conveyor-period-us> <disk-period-us>
RUNNING <host-session> <ESP-boot-id> <conveyor-period-us> <disk-period-us>
SORT <host-session> <ESP-boot-id> <sequence> GOOD
SORT <host-session> <ESP-boot-id> <sequence> BAD
ACK <host-session> <ESP-boot-id> <sequence> ACCEPTED
ACK <host-session> <ESP-boot-id> <sequence> DONE
PING <host-session> <ESP-boot-id>
PONG <host-session> <ESP-boot-id>
HOME <host-session> <ESP-boot-id>
```

The payload examples omit the computed checksum; sending plain `1` or `0` does
not move the servo. The checksum detects line corruption; it is not authentication.

ESP32 servo-only firmware omits `FEED_V1`; the Pi refuses to start motors without
that capability. `RUN` periods are 1000–100000 µs. A zero disables that motor for
individual commissioning; both zero is rejected. Repeating identical `RUN`
parameters acknowledges the existing run without restarting it. Changing speed
requires HOME and a new session. A different host cannot take over an active
session. `HOME` stops both motors, turns off lighting, requests servo neutral and
invalidates the session. It does not mechanically home the disk: no home sensor
is connected.

Commands are bound to a random host session and ESP boot ID. Sequence numbers
increase by one. A retry with the same sequence/class receives another ACK
without repeating movement; changing the class under the same sequence is an
error. There is one outstanding servo command at a time, while the Pi's FIFO
holds future seeds. The Pi sends heartbeats and polls completion during camera
processing, retrying lost replies with the same command ID only while enough
time remains before arrival. Same-class seeds have distinct IDs but do not move
the servo again. `DONE` is immediate for an already-held outlet.

Unexpected reboot, wrong session, serial failure, missing completion, stale
camera frames, invalid outputs or inference errors stop sorting. There is no
automatic reconnect/replay of an uncertain command. The ESP's nonblocking timer
acknowledges settling and keeps the last outlet position. Its watchdog
stops both motors, turns off lighting, requests neutral and invalidates the
session after 3 seconds without host contact. Motor waveforms also have a
3-second lease renewed by PING, so they expire if the main loop stalls while
interrupts still run. There is no automatic feed restart after a fault; clear
seeds from the inspection/transport area and restart with an empty view.

`DONE` means the commanded outlet has completed its settling delay (or was
already held); it is not sensor feedback. It does not prove that a servo moved
or a seed reached the correct outlet. `sort_events.jsonl` records `seed_queued`,
`servo_command` (or `dry_run_command`) and `seed_passed_estimate` events by seed ID.
The last event is a position estimate after confirmed servo readiness, not a
physical passage measurement. Monotonic timestamps are valid only within a run.

## Verification and references

Run the tests in an environment with the common dependencies plus pandas and
scikit-learn. A C++ compiler enables the native firmware simulation tests:

```bash
python -m unittest discover -s tests -v
```

Tests exercise color/range/EXIF preprocessing, class mapping, malformed model
outputs, metadata/model mismatch, uncertainty, seed latching, stale evidence,
CRC corruption, partial/oversized serial frames, retries, ESP reboot, completion
timeouts, exact-duplicate removal, grouped splitting, augmentation reruns,
calibration rejection, held good/bad positions, unchanged same-class PWM, watchdog,
motor session/rate checks, partial motor-start failure, feed retries and shutdown,
mechanical rate calculations, multiple FIFO seeds in transit, clearance deadlines,
uncertain-seed rejection, overflow, old-firmware rejection and concurrent camera processing,
and `millis()` rollover. The native firmware tests use stub board APIs; they do
not replace compilation with your selected Arduino board core or physical tests.
The integrated sketch was also compiled for `esp8266:esp8266:nodemcuv2` using
ESP8266 Arduino core 3.1.2. Physical direction, pulse integrity, driver current,
loaded startup and seed travel timing still require bench verification.

Training and deployment follow the
[Keras fine-tuning guidance](https://keras.io/guides/transfer_learning/),
[LiteRT interpreter contract](https://ai.google.dev/edge/api/tflite/python/tf/lite/Interpreter),
[Espressif LEDC APIs](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/ledc.html),
[ESP8266 per-pin waveforms](https://github.com/esp8266/Arduino/blob/3.1.2/cores/esp8266/core_esp8266_waveform.h),
and [pySerial timeout/access behavior](https://pyserial.readthedocs.io/en/stable/pyserial_api.html).

## Live sorting dashboard (website)

A browser dashboard that runs the newest model on a USB camera and counts good and bad seeds live.

```bash
./run_dashboard.sh              # first run installs Python deps and builds the site
./run_dashboard.sh --camera 1   # use the second camera (0 is usually the laptop's built-in one)
```

Open http://127.0.0.1:8000. You can also switch cameras from the dropdown on the page.

- **Model**: the newest timestamped run in `model_outputs/` (`maize_mobilenetv3large_final.keras`).
  Per-class thresholds are recomputed from that run's `threshold_search_results.csv` the same way the notebook does.
- **Counting**: uses the same `DecisionGate` as `main.py`. The camera must first see an empty view. Each seed is
  counted once, when the model is confident across 3 frames. The view must be empty again before the next seed.
  Objects that leave without a confident class are listed as *unsure* and are not added to either count.
- **Options**: `--roi X Y W H` crops the region the model looks at; `--host 0.0.0.0` lets phones on the same Wi-Fi view it.
- Every counted seed is appended to `web_sort_events.jsonl`.
- Frontend dev with hot reload: run the server, then `cd web/frontend && npm run dev`.
