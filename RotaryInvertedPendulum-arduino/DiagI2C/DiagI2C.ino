#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin();
  delay(500);
  Serial.println("AS5600 raw register read");
}

void loop() {
  // Write register pointer to STATUS (0x0B) — STOP instead of Repeated START
  Wire.beginTransmission(0x36);
  Wire.write(0x0B);
  byte rc = Wire.endTransmission(true);   // true = send STOP

  // Read 1 byte (new START)
  byte n_status = Wire.requestFrom((uint8_t)0x36, (uint8_t)1);
  byte status = Wire.read();

  // Write register pointer to RAW_ANGLE (0x0C) — STOP
  Wire.beginTransmission(0x36);
  Wire.write(0x0C);
  Wire.endTransmission(true);            // true = send STOP

  // Read 2 bytes (new START)
  byte n_raw = Wire.requestFrom((uint8_t)0x36, (uint8_t)2);
  byte hi = Wire.read();
  byte lo = Wire.read();
  uint16_t raw = ((uint16_t)hi << 8 | lo) & 0x0FFF;

  Serial.print("rc="); Serial.print(rc);
  Serial.print(" n="); Serial.print(n_status); Serial.print("/"); Serial.print(n_raw);
  Serial.print(" STATUS=0x"); Serial.print(status, HEX);
  Serial.print(" hi=0x"); Serial.print(hi, HEX);
  Serial.print(" lo=0x"); Serial.print(lo, HEX);
  Serial.print(" angle="); Serial.println(raw);

  delay(500);
}
