// AS5600 I2C backend: Hailege clone module.
//
// This clone's I2C behavior degrades badly under the standard AS5600
// library (Wire, interrupt-driven, 400 kHz) combined with FastAccelStepper's
// Timer1 ISR — see docs/BOM.md's "Amazon France (as-built sourcing)"
// listing ("behaves differently from the original Seeed module"). This
// backend instead:
//   - Recovers a possibly-stuck I2C bus at boot (9 SCL pulses + manual
//     STOP) — a reset mid-transaction can leave this module holding the
//     bus.
//   - Reads the RAW_ANGLE register via direct polling TWI register
//     access instead of the Wire library, bypassing its interrupt-driven
//     state machine entirely (eliminates sensitivity to Timer1 ISR
//     preemption), with retries.
//   - Runs the bus at 100 kHz instead of 400 kHz (400 kHz bits get
//     corrupted by the Timer1 ISR on this module).
//   - Skips the isConnected()/detectMagnet() startup gate: both rely on
//     an I2C write-only probe that's unreliable on this clone, while
//     direct RAW_ANGLE reads work correctly. The host validates state
//     values before engaging the motor.
//
// To use this profile: copy this file to ../hw_config.h (gitignored —
// see tools/pi_demo/README.md and flash_if_needed.py, which refuses to
// compile without it).

#define AS5600_I2C_ADDR  0x36
#define AS5600_RAW_REG   0x0C
#define AS5600_MAX_RETRIES 3

static void as5600_backend_setup(AS5600 &dev)
{
    // I²C bus recovery: always runs regardless of SDA state. After an
    // Arduino reset mid-transaction the AS5600 may be holding SDA LOW
    // (mid-byte) or HIGH (waiting for ACK) — both leave it unable to
    // respond to a fresh Wire.begin(). Nine SCL pulses let the slave
    // finish any in-progress byte. The STOP is generated cleanly by
    // driving SDA LOW while SCL is still LOW (end of last pulse), then
    // raising SCL HIGH, then raising SDA HIGH — avoiding the spurious
    // START that occurs if SDA goes LOW while SCL is already HIGH.
    {
        const uint8_t SDA_PIN = A4, SCL_PIN = A5;
        pinMode(SCL_PIN, OUTPUT);
        pinMode(SDA_PIN, INPUT_PULLUP);
        for (uint8_t i = 0; i < 9; i++) {
            digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
            digitalWrite(SCL_PIN, LOW);  delayMicroseconds(5);
        }
        // SCL is now LOW. Drive SDA LOW while SCL LOW (not a START).
        // Then raise SCL HIGH, then SDA HIGH = clean STOP condition.
        pinMode(SDA_PIN, OUTPUT);
        digitalWrite(SDA_PIN, LOW);  delayMicroseconds(5);
        digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
        digitalWrite(SDA_PIN, HIGH); delayMicroseconds(5);
        pinMode(SCL_PIN, INPUT);
        pinMode(SDA_PIN, INPUT);
        delay(10);
    }

    Wire.begin();
    Wire.setClock(100000);  // 100 kHz: 400 kHz bits (2.5 µs) get corrupted by
                            // FastAccelStepper Timer1 ISR; 100 kHz (10 µs/bit)
                            // is robust within the 2 ms sampleState() budget.
    dev.begin();
    // No startup gate on isConnected()/detectMagnet(): both rely on an I²C
    // write-only probe that is unreliable on this module clone, while
    // rawAngle() reads work correctly. The host validates state values before
    // engaging the motor.
}

static void as5600_backend_wait_magnet(AS5600 &dev)
{
    // No-op: see as5600_backend_setup()'s comment.
    (void)dev;
}

// Direct polling TWI read for AS5600 RAW_ANGLE register.
// Bypasses the Wire library's interrupt-driven state machine entirely,
// which eliminates sensitivity to Timer1 ISR preemption. Retries up to
// AS5600_MAX_RETRIES times (each retry adds ~100 µs for the NACK+delay).
// Masks Timer1 interrupts for the ~500 µs read window to avoid any
// preemption of the busy-wait loops; also suspends the Wire ISR so it
// doesn't race with this direct register access.
static bool as5600_backend_read(AS5600 &dev, long* out)
{
    (void)dev;
    uint8_t saved_timsk1 = TIMSK1;
    TIMSK1 = 0;

    uint8_t saved_twcr_twie = TWCR & _BV(TWIE);
    TWCR &= ~_BV(TWIE);

    bool ok = false;
    for (uint8_t attempt = 0; attempt < AS5600_MAX_RETRIES && !ok; attempt++)
    {
        if (attempt > 0) delayMicroseconds(150);

        uint16_t to;

        // START
        TWCR = _BV(TWINT) | _BV(TWSTA) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_START) goto stop_retry;

        // SLA+W
        TWDR = AS5600_I2C_ADDR << 1;
        TWCR = _BV(TWINT) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_MT_SLA_ACK) goto stop_retry;

        // Register address
        TWDR = AS5600_RAW_REG;
        TWCR = _BV(TWINT) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_MT_DATA_ACK) goto stop_retry;

        // Repeated START
        TWCR = _BV(TWINT) | _BV(TWSTA) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_REP_START) goto stop_retry;

        // SLA+R
        TWDR = (AS5600_I2C_ADDR << 1) | 1;
        TWCR = _BV(TWINT) | _BV(TWEN);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to || (TWSR & 0xF8) != TW_MR_SLA_ACK) goto stop_retry;

        // Read high byte (ACK)
        TWCR = _BV(TWINT) | _BV(TWEN) | _BV(TWEA);
        for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
        if (!to) goto stop_retry;
        {
            uint8_t hi = TWDR;
            // Read low byte (NACK — last byte)
            TWCR = _BV(TWINT) | _BV(TWEN);
            for (to = 20000; to && !(TWCR & _BV(TWINT)); to--);
            if (!to) goto stop_retry;
            uint8_t lo = TWDR;
            *out = ((long)hi << 8 | lo) & 0x0FFF;
            ok = true;
        }

        stop_retry:
        // Always send STOP to release the bus before next attempt or exit.
        TWCR = _BV(TWINT) | _BV(TWSTO) | _BV(TWEN);
        // Wait for STOP to complete (TWSTO clears itself when done).
        for (uint16_t w = 10000; w && (TWCR & _BV(TWSTO)); w--);
    }

    // Restore Wire ISR
    if (saved_twcr_twie) TWCR |= _BV(TWIE);
    TIMSK1 = saved_timsk1;
    return ok;
}
