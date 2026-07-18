// Diagnostic: check SCL and SDA levels BEFORE Wire.begin(), then attempt bus recovery
// with push-pull SCL (overrides slave clock stretching).
// After recovery, read raw angle via STOP (not repeated START).

#include <Wire.h>

#define SDA_PIN A4
#define SCL_PIN A5

void busRecovery() {
    // SDA as input so slave can drive bits; SCL as push-pull output to override stretching.
    pinMode(SDA_PIN, INPUT_PULLUP);
    pinMode(SCL_PIN, OUTPUT);

    // 9 SCL pulses — push-pull HIGH overrides slave clock stretching
    for (uint8_t i = 0; i < 9; i++) {
        digitalWrite(SCL_PIN, HIGH); delayMicroseconds(10);
        digitalWrite(SCL_PIN, LOW);  delayMicroseconds(10);
    }

    // SCL is now LOW. Generate clean STOP:
    //   drive SDA LOW while SCL LOW (not a START)
    //   then SCL HIGH, then SDA HIGH = STOP
    pinMode(SDA_PIN, OUTPUT);
    digitalWrite(SDA_PIN, LOW);  delayMicroseconds(10);
    digitalWrite(SCL_PIN, HIGH); delayMicroseconds(10);
    digitalWrite(SDA_PIN, HIGH); delayMicroseconds(10);

    // Release both pins, let Wire.begin() take over
    pinMode(SCL_PIN, INPUT);
    pinMode(SDA_PIN, INPUT);
    delay(20);
}

void setup() {
    // Step 1: read bus levels BEFORE anything (push-pull pullups just for reading)
    pinMode(SDA_PIN, INPUT_PULLUP);
    pinMode(SCL_PIN, INPUT_PULLUP);
    delayMicroseconds(10);
    bool sda0 = digitalRead(SDA_PIN);
    bool scl0 = digitalRead(SCL_PIN);
    pinMode(SDA_PIN, INPUT);
    pinMode(SCL_PIN, INPUT);

    // Step 2: bus recovery
    busRecovery();

    // Step 3: read bus levels AFTER recovery
    pinMode(SDA_PIN, INPUT_PULLUP);
    pinMode(SCL_PIN, INPUT_PULLUP);
    delayMicroseconds(10);
    bool sda1 = digitalRead(SDA_PIN);
    bool scl1 = digitalRead(SCL_PIN);
    pinMode(SDA_PIN, INPUT);
    pinMode(SCL_PIN, INPUT);

    // Step 4: Wire init
    Wire.begin();
    Wire.setClock(100000);
    delay(50);

    // Step 5: Serial + trigger
    Serial.begin(115200);
    delay(100);
    Serial.println("SEND_TRIGGER");
    while (Serial.available() == 0) {}
    Serial.read();

    Serial.println("=== DiagSCL ===");
    Serial.print("BEFORE: SDA="); Serial.print(sda0 ? "H" : "L");
    Serial.print(" SCL="); Serial.println(scl0 ? "H" : "L");
    Serial.print("AFTER:  SDA="); Serial.print(sda1 ? "H" : "L");
    Serial.print(" SCL="); Serial.println(scl1 ? "H" : "L");

    // Step 6: bare address probe
    Wire.beginTransmission(0x36);
    uint8_t rc0 = Wire.endTransmission();
    Serial.print("ADDR rc="); Serial.println(rc0);

    // Step 7: STOP-based register read
    Wire.beginTransmission(0x36);
    Wire.write(0x0C);
    uint8_t rc1 = Wire.endTransmission(true);  // STOP
    uint8_t n1  = Wire.requestFrom((uint8_t)0x36, (uint8_t)2);
    uint16_t raw1 = 0;
    if (n1 == 2) raw1 = ((uint16_t)Wire.read() << 8 | Wire.read()) & 0x0FFF;
    Serial.print("STOP rc="); Serial.print(rc1);
    Serial.print(" n="); Serial.print(n1);
    Serial.print(" raw="); Serial.println(raw1);
}

void loop() {
    delay(500);
    Wire.beginTransmission(0x36);
    Wire.write(0x0C);
    uint8_t rc = Wire.endTransmission(true);
    uint8_t n  = Wire.requestFrom((uint8_t)0x36, (uint8_t)2);
    uint16_t raw = 0;
    if (n == 2) raw = ((uint16_t)Wire.read() << 8 | Wire.read()) & 0x0FFF;
    Serial.print("rc="); Serial.print(rc);
    Serial.print(" n="); Serial.print(n);
    Serial.print(" raw="); Serial.println(raw);
}
