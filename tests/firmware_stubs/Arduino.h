#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#define OUTPUT 1
#define HIGH 1
#define LOW 0
extern uint32_t fakeNow;
extern std::vector<uint32_t> duties;
extern int pinLevels[17];
struct MockSerial {
  std::string incoming, outgoing;
  void begin(unsigned long) {}
  void print(const char* text) { outgoing += text; }
  int available() { return static_cast<int>(incoming.size()); }
  int read() { if (incoming.empty()) return -1; int c=incoming[0]; incoming.erase(0,1); return c; }
};
extern MockSerial Serial;
inline uint32_t millis() { return fakeNow; }
inline void yield() {}
inline bool ledcAttach(uint8_t, uint32_t, uint8_t) { return true; }
inline bool ledcWrite(uint8_t, uint32_t duty) { duties.push_back(duty); return true; }
inline double ledcSetup(uint8_t, double freq, uint8_t) { return freq; }
inline void ledcAttachPin(uint8_t, uint8_t) {}
inline void pinMode(uint8_t, int) {}
inline void digitalWrite(uint8_t pin, int value) { pinLevels[pin] = value; }
inline void analogWriteRange(uint32_t) {}
inline void analogWriteFreq(uint32_t) {}
inline void analogWrite(uint8_t, uint32_t duty) { duties.push_back(duty); }
