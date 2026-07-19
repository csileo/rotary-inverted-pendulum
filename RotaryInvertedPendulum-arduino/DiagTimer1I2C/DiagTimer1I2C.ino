// Diagnostic: does FastAccelStepper Timer1 corrupt AS5600 rawAngle()?
// Prints rawAngle() before and after engine.init() at 2 Mbaud.
// Expected: both should return non-zero (some angle 0-4095).
// If "after" is always 0 but "before" is not -> Timer1 is the cause.

#include <FastAccelStepper.h>
#include <AS5600.h>
#include <Wire.h>

#define STEP_PIN 9
#define DIR_PIN  2
#define ENABLE_PIN 5

FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
AS5600 as5600;

void setup() {
    Serial.begin(2000000);

    // I2C bus recovery: 9 SCL pulses to release a stuck SDA from a previous sketch.
    pinMode(A5, OUTPUT); pinMode(A4, OUTPUT);
    digitalWrite(A4, HIGH);
    for (int i = 0; i < 9; i++) {
        digitalWrite(A5, HIGH); delayMicroseconds(5);
        digitalWrite(A5, LOW);  delayMicroseconds(5);
    }
    // STOP condition
    digitalWrite(A5, HIGH); delayMicroseconds(5);
    digitalWrite(A4, HIGH);
    delayMicroseconds(5);
    pinMode(A5, INPUT); pinMode(A4, INPUT);

    delay(200);  // let AS5600 settle
    Wire.begin();
    Wire.setClock(100000);
    as5600.begin();

    // Read BEFORE engine.init()
    delay(200);
    uint16_t raw_before = as5600.rawAngle();
    Serial.print("rawAngle BEFORE engine.init(): ");
    Serial.println(raw_before);

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (stepper) {
        stepper->setDirectionPin(DIR_PIN);
        stepper->setEnablePin(ENABLE_PIN);
        stepper->setAutoEnable(false);
        stepper->setSpeedInUs(785);
        stepper->setForwardPlanningTimeInMs(8);
        stepper->disableOutputs();
    }

    // Read AFTER engine.init()
    delay(50);
    uint16_t raw_after = as5600.rawAngle();
    Serial.print("rawAngle AFTER  engine.init(): ");
    Serial.println(raw_after);
}

uint16_t readRawAngle_withSTOP() {
    // Standard mode: STOP between write and read (no Repeated START)
    Wire.beginTransmission(0x36);
    Wire.write(0x0C);  // RAW_ANGLE high byte register
    byte rc = Wire.endTransmission(true);  // STOP
    if (rc != 0) { Serial.print("STOP rc="); Serial.println(rc); return 0xFFFF; }
    byte n = Wire.requestFrom((uint8_t)0x36, (uint8_t)2);
    if (n != 2) { Serial.print("STOP n="); Serial.println(n); return 0xFFFF; }
    uint16_t raw = ((uint16_t)Wire.read() << 8 | Wire.read()) & 0x0FFF;
    return raw;
}

uint16_t readRawAngle_repeatedStart() {
    // Repeated START mode (what AS5600 library uses internally)
    Wire.beginTransmission(0x36);
    Wire.write(0x0C);
    byte rc = Wire.endTransmission(false);  // Repeated START
    if (rc != 0) { Serial.print("RS rc="); Serial.println(rc); return 0xFFFF; }
    byte n = Wire.requestFrom((uint8_t)0x36, (uint8_t)2);
    if (n != 2) { Serial.print("RS n="); Serial.println(n); return 0xFFFF; }
    uint16_t raw = ((uint16_t)Wire.read() << 8 | Wire.read()) & 0x0FFF;
    return raw;
}

void loop() {
    uint16_t lib   = as5600.rawAngle();
    uint16_t stop_  = readRawAngle_withSTOP();
    uint16_t rs_    = readRawAngle_repeatedStart();
    Serial.print("lib="); Serial.print(lib);
    Serial.print("  STOP="); Serial.print(stop_);
    Serial.print("  RS="); Serial.println(rs_);
    delay(500);
}
