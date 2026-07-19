// Diagnostic: does LowLevelServer's AS5600 polling routine (as5600_read_polling,
// which masks TIMSK1 during each read) interfere with FastAccelStepper's
// ability to actually step, when both run together the same way loop()
// does in LowLevelServer? DiagFastAccelOnly already proved FastAccelStepper
// alone works fine on this rig — this adds back only the I2C polling piece.
//
// No host commands needed: engages and commands a fixed accel ~0.7 s after
// boot, then samples I2C every 2 ms exactly like LowLevelServer's loop().
// Watch the motor and the periodic printout: pos should climb steadily if
// stepping isn't disturbed; i2c_ok/fail counts show whether the AS5600 is
// actually responding on this wiring.

#include <FastAccelStepper.h>
#include <Wire.h>
#include <util/twi.h>
#include <AS5600.h>

AS5600 as5600_lib;  // TEMP diagnostic: does calling begin() disturb the polling reads below?

#define DIR_PIN 2
#define STEP_PIN 9     // Timer1 OC1A
#define ENABLE_PIN 5

#define AS5600_I2C_ADDR  0x36
#define AS5600_RAW_REG   0x0C
#define AS5600_MAX_RETRIES 3

const uint16_t SAMPLE_PERIOD_US = 2000;
static uint32_t last_sample_us = 0;
static uint32_t i2c_ok_count = 0;
static uint32_t i2c_fail_count = 0;

FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;

// Verbatim copy of LowLevelServer's polling I2C read, to reproduce the
// exact same TIMSK1/TWCR manipulation and timing.
static bool as5600_read_polling(long* out)
{
    uint8_t saved_twcr_twie = TWCR & _BV(TWIE);
    TWCR &= ~_BV(TWIE);

    bool ok = false;
    for (uint8_t attempt = 0; attempt < AS5600_MAX_RETRIES && !ok; attempt++)
    {
        if (attempt > 0) delayMicroseconds(150);

        uint16_t to;

        TWCR = _BV(TWINT) | _BV(TWSTA) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_START) goto stop_retry;

        TWDR = AS5600_I2C_ADDR << 1;
        TWCR = _BV(TWINT) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_MT_SLA_ACK) goto stop_retry;

        TWDR = AS5600_RAW_REG;
        TWCR = _BV(TWINT) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_MT_DATA_ACK) goto stop_retry;

        TWCR = _BV(TWINT) | _BV(TWSTA) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_REP_START) goto stop_retry;

        TWDR = (AS5600_I2C_ADDR << 1) | 1;
        TWCR = _BV(TWINT) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_MR_SLA_ACK) goto stop_retry;

        TWCR = _BV(TWINT) | _BV(TWEN) | _BV(TWEA);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to) goto stop_retry;
        {
            uint8_t hi = TWDR;
            TWCR = _BV(TWINT) | _BV(TWEN);
            for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
            if (!to) goto stop_retry;
            uint8_t lo = TWDR;
            *out = ((long)hi << 8 | lo) & 0x0FFF;
            ok = true;
        }

        stop_retry:
        TWCR = _BV(TWINT) | _BV(TWSTO) | _BV(TWEN);
        for (uint16_t w = 10000; w && (TWCR & _BV(TWSTO)); w--);
    }

    if (saved_twcr_twie) TWCR |= _BV(TWIE);
    return ok;
}

void sampleI2CLikeLowLevelServer()
{
    uint8_t saved_timsk1 = TIMSK1;
    TIMSK1 = 0;
    long raw = 0;
    bool ok = as5600_read_polling(&raw);
    TIMSK1 = saved_timsk1;
    if (ok) i2c_ok_count++; else i2c_fail_count++;
}

void setup()
{
    Serial.begin(2000000);  // TEMP diagnostic: was 115200
    while (!Serial) { ; }

    // Same bus recovery as LowLevelServer.
    {
        const uint8_t SDA_PIN = A4, SCL_PIN = A5;
        pinMode(SCL_PIN, OUTPUT);
        pinMode(SDA_PIN, INPUT_PULLUP);
        for (uint8_t i = 0; i < 9; i++) {
            digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
            digitalWrite(SCL_PIN, LOW);  delayMicroseconds(5);
        }
        pinMode(SDA_PIN, OUTPUT);
        digitalWrite(SDA_PIN, LOW);  delayMicroseconds(5);
        digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
        digitalWrite(SDA_PIN, HIGH); delayMicroseconds(5);
        pinMode(SCL_PIN, INPUT);
        pinMode(SDA_PIN, INPUT);
        delay(10);
    }
    Wire.begin();
    Wire.setClock(100000);
    as5600_lib.begin();  // TEMP diagnostic

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        Serial.println("FATAL: stepperConnectToPin failed");
        while (true) { ; }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);
    stepper->setAutoEnable(false);
    stepper->setSpeedInUs(550);
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    delay(500);
    Serial.println("Enabling outputs...");
    stepper->enableOutputs();
    delay(200);

    // TEMP diagnostic: replicate the exact ENGAGE-then-SET_ACCEL double call
    // sequence (hold at 0, then command real accel) instead of a single
    // direct call, while keeping the I2C polling that already proved fine.
    Serial.println("moveByAcceleration(0, true)  -- mimics CMD_ENGAGE_MOTOR");
    stepper->moveByAcceleration(0, true);
    delay(300);
    Serial.println("moveByAcceleration(5000, true) -- mimics first CMD_SET_ACCEL");
    stepper->moveByAcceleration(5000, true);

    last_sample_us = micros();
}

void loop()
{
    uint32_t now_us = micros();
    if ((uint32_t)(now_us - last_sample_us) >= SAMPLE_PERIOD_US)
    {
        last_sample_us = now_us;
        sampleI2CLikeLowLevelServer();
    }

    static uint32_t last_print = 0;
    if (millis() - last_print > 200)
    {
        last_print = millis();
        Serial.print("pos="); Serial.print(stepper->getCurrentPosition());
        Serial.print("  speed="); Serial.print(stepper->getCurrentSpeedInMilliHz());
        Serial.print("  i2c_ok="); Serial.print(i2c_ok_count);
        Serial.print("  i2c_fail="); Serial.println(i2c_fail_count);
    }
}
