#pragma once
#include "Arduino.h"
struct Wave { uint32_t high = 0, low = 0, lease = 0; unsigned updates = 0; };
extern Wave waves[17];
extern int failingWavePin;
inline int startWaveform(uint8_t pin, uint32_t high, uint32_t low, uint32_t lease = 0) {
  if (pin == failingWavePin) return false;
  waves[pin].high = high; waves[pin].low = low; waves[pin].lease = lease;
  ++waves[pin].updates;
  if (pin == 4) duties.push_back(high);
  return true;
}
inline int stopWaveform(uint8_t pin) { waves[pin].high = waves[pin].low = 0; return true; }
