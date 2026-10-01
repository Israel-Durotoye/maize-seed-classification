/* Single-seed sorter: Raspberry Pi <-> USB serial <-> ESP32/ESP8266.
 * Integrated feeder on ESP8266 core 3.1.2; ESP32 retains servo-only support.
 * No external servo library required. See README.md before connecting power.
 * Commands are CRC16-CCITT checked, session/boot bound, and sequence deduplicated.
 */
#include <Arduino.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#if defined(ARDUINO_ARCH_ESP32)
#include <esp_system.h>
#include <esp_arduino_version.h>
#elif defined(ARDUINO_ARCH_ESP8266)
#include <core_esp8266_waveform.h>
extern "C" {
#include <user_interface.h>
}
#else
#error "Select an ESP32-family or ESP8266 board."
#endif

// GPIO numbers, NOT the board's printed Dx numbers. Change for your actual board.
constexpr uint8_t SERVO_PIN = 5; // D1, proven physical wiring
constexpr int GOOD_ANGLE = 45;
constexpr int BAD_ANGLE = 135;
constexpr int NEUTRAL_ANGLE = 90;
// Conservative pulse range; calibrate unloaded against your servo's datasheet.
constexpr uint32_t SERVO_MIN_US = 1000;
constexpr uint32_t SERVO_MAX_US = 2000;
constexpr uint32_t SERVO_MOVE_MS = 350; // Calibrate worst-case 45 <-> 135 degree travel under load.
constexpr uint32_t LINK_TIMEOUT_MS = 3000;
constexpr uint32_t BAUD = 115200;
constexpr uint8_t PWM_BITS = 14;
constexpr uint32_t PWM_MAX = (1UL << PWM_BITS) - 1;
// ESP8266 NodeMCU/D1-mini labels. DIR pins connect ONLY to driver DIR inputs.
constexpr uint8_t CONVEYOR_STEP_PIN = 4;  // D2
constexpr uint8_t CONVEYOR_DIR_PIN = 16;  // D0 dummy; real belt DIR remains tied to 3.3 V
constexpr uint8_t DISK_STEP_PIN = 14;     // D5
constexpr uint8_t DISK_DIR_PIN = 12;      // D6, physically connected
constexpr uint8_t LIGHT_PIN = 13;         // D7, active-high MOSFET gate
constexpr uint8_t CONVEYOR_DIRECTION = HIGH;
constexpr uint8_t DISK_DIRECTION = HIGH;
constexpr uint32_t STEP_HIGH_US = 10;
constexpr uint32_t MIN_STEP_PERIOD_US = 250;    // Up to 4000 pulses/s; proven setup uses ~1599.
constexpr uint32_t MAX_STEP_PERIOD_US = 100000; // 10 pulses/s.
uint32_t conveyorPeriod = 0, diskPeriod = 0;
bool feedRunning = false;
uint32_t lastFeedRenewal = 0;
static_assert(GOOD_ANGLE >= 0 && GOOD_ANGLE <= 180 && BAD_ANGLE >= 0 && BAD_ANGLE <= 180, "Invalid servo angles");
static_assert(NEUTRAL_ANGLE >= 0 && NEUTRAL_ANGLE <= 180, "Invalid neutral angle");
static_assert(SERVO_MIN_US >= 500 && SERVO_MAX_US <= 2500 && SERVO_MIN_US < SERVO_MAX_US, "Invalid pulse limits");
static_assert(SERVO_MOVE_MS >= 50 && SERVO_MOVE_MS <= 3000, "Invalid servo movement timing");

enum Phase { IDLE, MOVING };
Phase phase = IDLE;
uint32_t phaseStarted = 0;
uint32_t lastContact = 0;
uint32_t lastSequence = 0;
bool sessionActive = false;
bool pwmHealthy = false;
char hostSession[17] = "";
char bootId[9] = "";
char lastClass[5] = "";
char currentClass[5] = ""; // Empty means neutral; GOOD/BAD remain held until changed or homed.
char inputLine[193];
size_t inputLength = 0;
bool discardLine = false;

uint16_t crc16(const char* text) {
  uint16_t crc = 0xFFFF;
  while (*text) {
    crc ^= static_cast<uint16_t>(static_cast<uint8_t>(*text++)) << 8;
    for (uint8_t i = 0; i < 8; ++i)
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
  }
  return crc;
}

void reply(const char* payload) {
  char frame[224];
  snprintf(frame, sizeof(frame), "%s|%04X\n", payload, crc16(payload));
  Serial.print(frame);
}

bool isHex(const char* value, size_t length) {
  if (strlen(value) != length) return false;
  for (size_t i = 0; i < length; ++i)
    if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'A' && value[i] <= 'F'))) return false;
  return true;
}

bool writeAngle(int angle) {
  const uint32_t pulse = SERVO_MIN_US + ((SERVO_MAX_US - SERVO_MIN_US) * angle) / 180;
#if defined(ARDUINO_ARCH_ESP32)
  const uint32_t duty = (pulse * PWM_MAX + 10000) / 20000;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  return ledcWrite(SERVO_PIN, duty);
#else
  ledcWrite(0, duty);
  return true;
#endif
#else
  // Independent period per pin: no global analogWrite frequency/range changes.
  return startWaveform(SERVO_PIN, pulse, 20000 - pulse, 0);
#endif
}

void stopFeed() {
#if defined(ARDUINO_ARCH_ESP8266)
  stopWaveform(CONVEYOR_STEP_PIN);
  stopWaveform(DISK_STEP_PIN);
  digitalWrite(CONVEYOR_STEP_PIN, LOW);
  digitalWrite(DISK_STEP_PIN, LOW);
#endif
  feedRunning = false;
  conveyorPeriod = diskPeriod = 0;
}

void setLight(bool enabled) {
#if defined(ARDUINO_ARCH_ESP8266)
  digitalWrite(LIGHT_PIN, enabled ? HIGH : LOW); // Constant light; no camera PWM banding.
#else
  (void)enabled;
#endif
}

bool refreshFeed() {
#if defined(ARDUINO_ARCH_ESP8266)
  // Finite waveforms expire even if loop() stalls. PING renews the lease.
  // Updating an existing waveform preserves its phase rather than restarting it.
  const uint32_t lease = LINK_TIMEOUT_MS * 1000;
  if (conveyorPeriod && !startWaveform(CONVEYOR_STEP_PIN, STEP_HIGH_US,
                                     conveyorPeriod - STEP_HIGH_US, lease)) return false;
  if (diskPeriod && !startWaveform(DISK_STEP_PIN, STEP_HIGH_US,
                                 diskPeriod - STEP_HIGH_US, lease)) return false;
  lastFeedRenewal = millis();
  return true;
#else
  return false;
#endif
}

void stopAndHome() {
  stopFeed();
  setLight(false);
  writeAngle(NEUTRAL_ANGLE);
  currentClass[0] = '\0';
  phase = IDLE;
  sessionActive = false;  // A new handshake is required after a fault or HOME.
}

void acknowledge(const char* state) {
  char payload[100];
  snprintf(payload, sizeof(payload), "ACK %s %s %lu %s", hostSession, bootId,
           static_cast<unsigned long>(lastSequence), state);
  reply(payload);
}

void ready() {
  char payload[100];
#if defined(ARDUINO_ARCH_ESP8266)
  const char* capabilities = " FEED_V1 HOLD_V2";
#else
  const char* capabilities = " HOLD_V2";
#endif
  snprintf(payload, sizeof(payload), "READY %s %s %lu 0%s", hostSession, bootId,
           static_cast<unsigned long>(SERVO_MOVE_MS), capabilities);
  reply(payload);
}

bool parsePeriod(const char* value, uint32_t& period) {
  if (!*value) return false;
  for (const char* c = value; *c; ++c) if (*c < '0' || *c > '9') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long parsed = strtoul(value, &end, 10);
  if (errno || *end || parsed > MAX_STEP_PERIOD_US ||
      (parsed && parsed < MIN_STEP_PERIOD_US)) return false;
  period = static_cast<uint32_t>(parsed);
  return true;
}

void handleLine(char* line) {
  char* separator = strrchr(line, '|');
  if (!separator || !isHex(separator + 1, 4)) return;
  const uint16_t expected = static_cast<uint16_t>(strtoul(separator + 1, nullptr, 16));
  *separator = '\0';
  if (crc16(line) != expected) return; // Corrupted lines cannot move a servo.
  char* fields[7];
  size_t count = 0;
  char* context = nullptr;
  char* token = strtok_r(line, " ", &context);
  while (token && count < 7) {
    fields[count++] = token;
    token = strtok_r(nullptr, " ", &context);
  }
  if (!count || token) return;
  if (!pwmHealthy) { reply("FAULT PWM"); return; }

  if (strcmp(fields[0], "HELLO") == 0) {
    if (count != 2 || !isHex(fields[1], 16)) { reply("ERR FORMAT"); return; }
    // Retried handshake from the SAME host must never reset the sequence counter.
    if (sessionActive && strcmp(fields[1], hostSession) == 0) {
      lastContact = millis(); ready(); return;
    }
    if (phase != IDLE || sessionActive) { reply("ERR BUSY"); return; }
    // Do not permit replay of a completed/abandoned session after HOME or timeout.
    if (strcmp(fields[1], hostSession) == 0) { reply("ERR NEW_SESSION_REQUIRED"); return; }
    strncpy(hostSession, fields[1], sizeof(hostSession));
    lastSequence = 0;
    lastClass[0] = '\0';
    sessionActive = true;
    lastContact = millis();
    setLight(true);
    ready();
    return;
  }
  if (count < 3 || !sessionActive || strcmp(fields[1], hostSession) != 0 || strcmp(fields[2], bootId) != 0) {
    reply("ERR SESSION"); return;
  }
  if (strcmp(fields[0], "PING") == 0 && count == 3) {
    lastContact = millis();
    if (feedRunning && !refreshFeed()) {
      stopAndHome(); reply("FAULT FEED"); return;
    }
    char payload[80];
    snprintf(payload, sizeof(payload), "PONG %s %s", hostSession, bootId);
    reply(payload);
    return;
  }
  if (strcmp(fields[0], "HOME") == 0 && count == 3) {
    stopAndHome(); reply("HOMED"); return;
  }
  if (strcmp(fields[0], "RUN") == 0 && count == 5) {
#if !defined(ARDUINO_ARCH_ESP8266)
    reply("ERR UNSUPPORTED"); return;
#else
    uint32_t belt = 0, disk = 0;
    if (!parsePeriod(fields[3], belt) || !parsePeriod(fields[4], disk) || (!belt && !disk)) {
      reply("ERR RATE"); return;
    }
    if (feedRunning && (belt != conveyorPeriod || disk != diskPeriod)) {
      reply("ERR CONFLICT"); return; // Stop and open a new session to change speeds.
    }
    if (phase != IDLE) { reply("ERR BUSY"); return; }
    lastContact = millis();
    if (!feedRunning) {
      conveyorPeriod = belt;
      diskPeriod = disk;
      if (!refreshFeed()) { stopAndHome(); reply("FAULT FEED"); return; }
      feedRunning = true;
    }
    char payload[100];
    snprintf(payload, sizeof(payload), "RUNNING %s %s %lu %lu", hostSession, bootId,
             static_cast<unsigned long>(conveyorPeriod), static_cast<unsigned long>(diskPeriod));
    reply(payload);
    return;
#endif
  }
  if (strcmp(fields[0], "SORT") != 0 || count != 5) { reply("ERR FORMAT"); return; }
  if (strcmp(fields[4], "GOOD") != 0 && strcmp(fields[4], "BAD") != 0) { reply("ERR CLASS"); return; }
  for (const char* c = fields[3]; *c; ++c)
    if (*c < '0' || *c > '9') { reply("ERR SEQUENCE"); return; }
  errno = 0;
  char* end = nullptr;
  unsigned long parsed = strtoul(fields[3], &end, 10);
  if (errno || *end || !parsed || parsed > UINT32_MAX) { reply("ERR SEQUENCE"); return; }
  const uint32_t sequence = static_cast<uint32_t>(parsed);
  lastContact = millis();
  if (sequence == lastSequence) {
    if (strcmp(fields[4], lastClass) != 0) { reply("ERR CONFLICT"); return; }
    acknowledge(phase == IDLE ? "DONE" : "ACCEPTED");
    return; // Duplicate command: acknowledge, never restart the movement.
  }
  if (lastSequence == UINT32_MAX || sequence != lastSequence + 1) { reply("ERR SEQUENCE"); return; }
  if (phase != IDLE) { reply("ERR BUSY"); return; }
  lastSequence = sequence;
  strncpy(lastClass, fields[4], sizeof(lastClass));
  if (strcmp(lastClass, currentClass) == 0) {
    acknowledge("DONE");
    return; // Requested outlet is already held: record sequence, but do not move.
  }
  if (!writeAngle(strcmp(lastClass, "GOOD") == 0 ? GOOD_ANGLE : BAD_ANGLE)) {
    pwmHealthy = false; stopAndHome(); reply("FAULT PWM"); return;
  }
  strncpy(currentClass, lastClass, sizeof(currentClass));
  phase = MOVING;
  phaseStarted = millis();
  acknowledge("ACCEPTED");
}

void updateServo() {
  const uint32_t now = millis();
  // Never restart an expired waveform when loop() resumes after a stall.
  if (feedRunning && static_cast<uint32_t>(now - lastFeedRenewal) >= LINK_TIMEOUT_MS) {
    stopAndHome(); reply("FAULT FEED_LEASE"); return;
  }
  if (sessionActive && static_cast<uint32_t>(now - lastContact) > LINK_TIMEOUT_MS) {
    stopAndHome(); reply("FAULT LINK_TIMEOUT"); return;
  }
  if (phase == MOVING && static_cast<uint32_t>(now - phaseStarted) >= SERVO_MOVE_MS) {
    phase = IDLE;
    acknowledge("DONE"); // Position remains held; no position-feedback sensor is connected.
  }
}

void setup() {
#if defined(ARDUINO_ARCH_ESP8266)
  // Set output latches before changing pin mode, with motors stopped at boot.
  digitalWrite(CONVEYOR_STEP_PIN, LOW); pinMode(CONVEYOR_STEP_PIN, OUTPUT);
  digitalWrite(DISK_STEP_PIN, LOW); pinMode(DISK_STEP_PIN, OUTPUT);
  digitalWrite(CONVEYOR_DIR_PIN, CONVEYOR_DIRECTION); pinMode(CONVEYOR_DIR_PIN, OUTPUT);
  digitalWrite(DISK_DIR_PIN, DISK_DIRECTION); pinMode(DISK_DIR_PIN, OUTPUT);
  digitalWrite(LIGHT_PIN, LOW); pinMode(LIGHT_PIN, OUTPUT);
#endif
  Serial.begin(BAUD);
#if defined(ARDUINO_ARCH_ESP32)
  snprintf(bootId, sizeof(bootId), "%08lX", static_cast<unsigned long>(esp_random()));
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  pwmHealthy = ledcAttach(SERVO_PIN, 50, PWM_BITS);
#else
  pwmHealthy = ledcSetup(0, 50, PWM_BITS) > 0;
  if (pwmHealthy) ledcAttachPin(SERVO_PIN, 0);
#endif
#else
  snprintf(bootId, sizeof(bootId), "%08lX", static_cast<unsigned long>(os_random()));
  pinMode(SERVO_PIN, OUTPUT);
  pwmHealthy = true;
#endif
  if (pwmHealthy) pwmHealthy = writeAngle(NEUTRAL_ANGLE);
  char payload[32];
  snprintf(payload, sizeof(payload), "BOOT %s", bootId);
  reply(pwmHealthy ? payload : "FAULT PWM");
}

void loop() {
  updateServo();
  // Bound work per pass: even a noisy serial sender cannot starve the servo timer.
  for (uint8_t n = 0; n < 64 && Serial.available(); ++n) {
    const int incoming = Serial.read();
    if (incoming < 0) break;
    const char c = static_cast<char>(incoming);
    if (c == '\n') {
      if (!discardLine) {
        inputLine[inputLength] = '\0';
        handleLine(inputLine);
      }
      inputLength = 0;
      discardLine = false;
    } else if (c != '\r' && !discardLine) {
      if (c < 32 || c > 126 || inputLength >= sizeof(inputLine) - 1) {
        discardLine = true;
        inputLength = 0;
      } else {
        inputLine[inputLength++] = c;
      }
    }
  }
  yield();
}
