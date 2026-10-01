#include "Arduino.h"
#include <cassert>
#include <iostream>
uint32_t fakeNow = 0;
std::vector<uint32_t> duties;
int pinLevels[17] = {};
MockSerial Serial;
#include "core_esp8266_waveform.h"
Wave waves[17];
int failingWavePin = -1;
#include "../esp32_maize_sorter/esp32_maize_sorter.ino"

void sendPayload(const std::string& payload) {
  char checksum[8]; snprintf(checksum,sizeof(checksum),"|%04X\n",crc16(payload.c_str()));
  Serial.incoming += payload + checksum;
  while (Serial.available()) loop();
}
void advance(uint32_t elapsed) { fakeNow += elapsed; loop(); }
int main() {
  setup();
  assert(!feedRunning);
  assert(pinLevels[LIGHT_PIN] == LOW);
  assert(crc16("123456789") == 0x29B1);
  assert(duties.size()==1 && phase==IDLE);
  const auto neutral = duties.back();
  const std::string host="AAAAAAAAAAAAAAAA", boot="12345678";
  const std::string prefix="SORT "+host+" "+boot+" ";
  sendPayload(prefix+"1 GOOD"); assert(duties.size()==1); // No handshake.
  sendPayload("HELLO "+host); assert(sessionActive);
  sendPayload(prefix+"1 GOOD"); assert(phase==MOVING && duties.size()==2 && duties.back()<neutral);
  sendPayload(prefix+"1 GOOD"); assert(duties.size()==2); // Lost ACK retransmission.
  sendPayload(prefix+"1 BAD"); assert(duties.size()==2);  // Same ID, different command rejected.
  sendPayload(prefix+"2 BAD"); assert(duties.size()==2);  // Busy: no queued actuation.
  sendPayload("HELLO "+host); assert(lastSequence==1);  // Retried HELLO preserves deduplication.
  advance(SETTLE_MS); assert(phase==IDLE && duties.back()<neutral);
  sendPayload(prefix+"1 GOOD"); assert(duties.size()==2); // Lost DONE never re-actuates.
  sendPayload(prefix+"2 GOOD"); assert(phase==IDLE && duties.size()==2); // Next GOOD: no motion.
  // Keep heartbeats alive well beyond the former neutral-return cycle.
  for (int i=0; i<12; ++i) {
    sendPayload("PING "+host+" "+boot); advance(500);
    assert(phase==IDLE && duties.size()==2 && duties.back()<neutral);
  }
  sendPayload(prefix+"3 BAD"); assert(phase==MOVING && duties.back()>neutral);
  advance(SETTLE_MS); assert(phase==IDLE && duties.size()==3 && duties.back()>neutral);
  const auto count=duties.size();
  sendPayload(prefix+"1 GOOD"); // Old sequence rejected.
  sendPayload(prefix+"4 UNKNOWN");
  sendPayload(prefix+"4294967296 GOOD");
  sendPayload("SORT "+host+" DEADBEEF 4 GOOD");
  Serial.incoming=prefix+"4 GOOD|0000\n";
  while (Serial.available()) loop();
  assert(duties.size()==count);
  Serial.incoming=std::string(250,'x')+"\n";
  while (Serial.available()) loop();
  sendPayload(prefix+"4 GOOD"); assert(duties.size()==count+1);
  advance(LINK_TIMEOUT_MS+1); assert(!sessionActive && phase==IDLE && duties.back()==neutral);
  const auto stoppedCount=duties.size();
  sendPayload(prefix+"4 GOOD"); sendPayload("HELLO "+host); assert(!sessionActive && duties.size()==stoppedCount);
  // millis() rollover must not break settling or cause a neutral return.
  fakeNow=UINT32_MAX-100;
  sendPayload("HELLO BBBBBBBBBBBBBBBB"); assert(sessionActive);
  sendPayload("PING BBBBBBBBBBBBBBBB "+boot);
  sendPayload("SORT BBBBBBBBBBBBBBBB "+boot+" 1 BAD");
  advance(SETTLE_MS); assert(phase==IDLE && duties.back()>neutral);
  sendPayload("HOME BBBBBBBBBBBBBBBB "+boot); assert(!sessionActive);
#if defined(ARDUINO_ARCH_ESP8266)
  // Motor startup is explicit and session-bound; lighting follows the session.
  const std::string feedHost = "CCCCCCCCCCCCCCCC";
  const std::string run = "RUN " + feedHost + " " + boot + " ";
  sendPayload(run + "1963 6563"); assert(!feedRunning);
  sendPayload("HELLO " + feedHost);
  assert(pinLevels[LIGHT_PIN] == HIGH && !feedRunning);
  assert(pinLevels[CONVEYOR_DIR_PIN] == HIGH && pinLevels[DISK_DIR_PIN] == HIGH);
  for (const auto& bad : {"0 0", "999 6563", "1963 100001", "-1 6563", "1963 nan", "4294967296 6563"}) {
    sendPayload(run + bad); assert(!feedRunning);
  }
  sendPayload(run + "1963 6563"); assert(feedRunning);
  assert(waves[CONVEYOR_STEP_PIN].high == 10 && waves[CONVEYOR_STEP_PIN].low == 1953);
  assert(waves[DISK_STEP_PIN].high == 10 && waves[DISK_STEP_PIN].low == 6553);
  assert(waves[CONVEYOR_STEP_PIN].lease == LINK_TIMEOUT_MS * 1000);
  const auto starts = waves[CONVEYOR_STEP_PIN].updates;
  sendPayload(run + "1963 6563"); assert(waves[CONVEYOR_STEP_PIN].updates == starts);
  sendPayload(run + "2000 6563"); assert(conveyorPeriod == 1963); // No live rate change.
  sendPayload("HELLO DDDDDDDDDDDDDDDD"); assert(std::string(hostSession) == feedHost);
  sendPayload("PING " + feedHost + " " + boot);
  assert(waves[CONVEYOR_STEP_PIN].updates == starts + 1); // Heartbeat renews lease.
  sendPayload("SORT " + feedHost + " " + boot + " 1 GOOD");
  assert(waves[SERVO_PIN].high + waves[SERVO_PIN].low == 20000);
  advance(SETTLE_MS); assert(feedRunning && duties.back()<neutral);
  advance(LINK_TIMEOUT_MS + 1);
  assert(!feedRunning && !sessionActive && pinLevels[LIGHT_PIN] == LOW);
  assert(waves[CONVEYOR_STEP_PIN].high == 0 && waves[DISK_STEP_PIN].high == 0);
  sendPayload(run + "1963 6563"); assert(!feedRunning); // Old session cannot restart.
  sendPayload("HELLO DDDDDDDDDDDDDDDD");
  sendPayload("RUN DDDDDDDDDDDDDDDD " + boot + " 0 6563");
  assert(feedRunning && waves[CONVEYOR_STEP_PIN].high == 0 && waves[DISK_STEP_PIN].high == 10);
  sendPayload("HOME DDDDDDDDDDDDDDDD " + boot);
  assert(!feedRunning && waves[DISK_STEP_PIN].high == 0);
  sendPayload("HELLO EEEEEEEEEEEEEEEE");
  sendPayload("RUN EEEEEEEEEEEEEEEE " + boot + " 1963 6563");
  advance(LINK_TIMEOUT_MS - 1);
  lastContact = fakeNow; // Other contact must not hide an expired pulse lease.
  advance(1);
  assert(!feedRunning && !sessionActive);
  sendPayload("PING EEEEEEEEEEEEEEEE " + boot); assert(!feedRunning);
  sendPayload("HELLO FFFFFFFFFFFFFFFF");
  failingWavePin = DISK_STEP_PIN;
  sendPayload("RUN FFFFFFFFFFFFFFFF " + boot + " 1963 6563");
  assert(!feedRunning && !sessionActive && waves[CONVEYOR_STEP_PIN].high == 0);
#else
  sendPayload("HELLO CCCCCCCCCCCCCCCC");
  sendPayload("RUN CCCCCCCCCCCCCCCC " + boot + " 1963 6563");
  assert(!feedRunning && Serial.outgoing.find("ERR UNSUPPORTED") != std::string::npos);
#endif
  std::cout << "PASS: firmware protocol, deduplication, held positions, directions, timers, watchdog, malformed input and rollover\n";
}
