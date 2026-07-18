#include <FastAccelStepper.h>
#include <AS5600.h>
#include <Wire.h>
#include <util/twi.h>

#include "StepperUtils.h"

// Direct polling TWI read for AS5600 RAW_ANGLE register.
// Bypasses the Wire library's interrupt-driven state machine entirely,
// which eliminates sensitivity to Timer1 ISR preemption. Retries up to
// AS5600_MAX_RETRIES times (each retry adds ~100 µs for the NACK+delay).
#define AS5600_I2C_ADDR  0x36
#define AS5600_RAW_REG   0x0C
#define AS5600_MAX_RETRIES 3

static bool as5600_read_polling(long* out)
{
    // Suspend Wire ISR so it doesn't race with our direct register access.
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
    return ok;
}

// Communication speed
const long BAUD_RATE = 2000000;

// Command bytes
#define CMD_READY 0x01
#define CMD_GET_STATE 0x02
#define CMD_SET_ACCEL 0x03   // was CMD_SET_TARGET (position-mode); now angular accel (rad/s²)
#define CMD_ENGAGE_MOTOR 0x04
#define CMD_DISENGAGE_MOTOR 0x05
#define CMD_TARE_PENDULUM 0x06   // re-zero pen_position_rad to current AS5600 reading
#define CMD_DEBUG_I2C     0x07   // diagnostic: raw I2C read, returns 4 bytes [rc, n, hi, lo]

// Pin assignments. STEP must be on pin 9 (Timer1 OC1A on ATmega328) for
// FastAccelStepper. DIR and ENABLE can be any digital pin.
#define DIR_PIN 2
#define STEP_PIN 9
#define ENABLE_PIN 5

// Accel-mode envelope. See pendulum_env.py for the corresponding sim
// constants. The velocity cap below corresponds to MAX_VELOCITY_RAD_S
// = 5 rad/s: 5 × (1600 steps/rev / 2π) ≈ 1273 steps/s ⇒ ~785 µs/step.
const uint32_t MOTOR_MIN_STEP_US = 550;  // ≈ 7 rad/s (was 785 ≈ 5 rad/s)

// Position safety limit (matches MOTOR_SAFE_LIMIT_RAD on the Python side,
// ±125°). Past the rail the firmware actively brakes (commands a fixed
// opposing accel) so the motor decelerates even if the host stops
// sending commands (USB hiccup, host hang). Clamping to zero instead
// would just let moveByAcceleration(0, true) coast the motor past the
// rail at peak velocity.
const int32_t MOTOR_SAFE_LIMIT_STEPS = (int32_t)((125.0f * PI / 180.0f) *
                                                  (1600.0f / (2.0f * PI)));
// Brake authority when past the rail. 150 rad/s² matches the
// pendulum_env.py MAX_ACCEL_RAD_S2 — strong enough to bleed off the
// 5 rad/s vel cap within ~33 ms.
const int32_t MOTOR_BRAKE_ACCEL_STEPS_S2 =
    (int32_t)(150.0f * (1600.0f / (2.0f * PI)));

// Encoder samples are kept in a ring buffer updated at 500 Hz; GET_STATE
// returns velocity computed as (newest - oldest)/Δt over a window of 5
// samples = 4 inter-sample gaps = 8 ms. Window halved from 10 → 5 on
// 2026-05-20 to cut ~9 ms of observation lag — the largest tunable
// component of the rig's policy-loop lag. Tradeoff: ~√2× noisier
// per-sample velocity estimate. See docs/transport_delay.md.
const uint16_t SAMPLE_PERIOD_US = 2000;
const uint8_t  BUFFER_SIZE      = 16;
const uint8_t  VEL_WINDOW       = 5;
// Discard impossibly-large per-sample wraps as I²C glitches. Real
// pendulum tops out at ~50 rad/s, so in one 2 ms sample period the
// raw AS5600 reading can change by at most 50·0.002·4096/(2π) ≈ 65 LSB.
// Anything more is almost certainly a corrupted I²C transaction —
// keep the previous reading rather than letting it pollute the
// accumulator with a spurious ±2π wrap.
const long PEN_RAW_MAX_DELTA_LSB = 500;

static int32_t motor_step_buf[BUFFER_SIZE];   // raw stepper position (steps)
static float   pen_rad_buf[BUFFER_SIZE];      // accumulated pendulum angle (rad)
static uint32_t time_us_buf[BUFFER_SIZE];     // sample timestamps (µs)
static uint8_t  buf_head = 0;                 // next write index
static bool     buf_filled = false;           // becomes true after first full lap
static uint32_t last_sample_us = 0;

// Continuously-tracked pendulum angle (independent of GET_STATE cadence so
// rapid wraparounds aren't missed).
static long    pen_raw_prev   = -1;           // -1 = first read sentinel
static float   pen_position_rad = 0.0f;

// State variables
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
AS5600 as5600;

// `motor_engaged` is only touched from loop() / handleCommand() — no ISR
// access — so `volatile` would only mislead future readers. Plain bool.
bool motor_engaged = false;

// Host-liveness watchdog. Without it, a killed/hung host process leaves
// moveByAcceleration() applying the LAST commanded accel forever — the
// motor winds up to the velocity cap and stalls against the mechanical
// hard stop under full torque (incident 2026-07-05: fried the Nano's
// CH340 and the host PC's USB-port ESD protection). The position-limit
// brake in CMD_SET_ACCEL can't help: it only runs when a command
// arrives.
//
// Armed by the first CMD_SET_ACCEL after an engage (so the idle gap
// between CMD_ENGAGE_MOTOR and the control loop's first tick can't
// false-trigger), refreshed by every received command, disarmed on
// engage/disengage. When armed and no command has arrived for
// WATCHDOG_TIMEOUT_US, loop() force-stops and disengages the motor —
// same path as CMD_DISENGAGE_MOTOR. The host re-engages on the next
// episode/run start, so recovery is automatic.
//
// 200 ms = 7-10 consecutive missed ticks at the 35-50 Hz control rates
// this rig uses — far beyond any legitimate host pause during an active
// episode, far shorter than a wind-up into the hard stop matters.
const uint32_t WATCHDOG_TIMEOUT_US = 200000UL;
static uint32_t last_cmd_us = 0;
static bool watchdog_armed = false;

// Function prototypes
void handleCommand();
void sendState();
void sampleState();
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s);

void setup()
{
    Serial.begin(BAUD_RATE);

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
    as5600.begin();
    // No startup gate on isConnected()/detectMagnet(): both rely on an I²C
    // write-only probe that is unreliable on this module clone, while
    // rawAngle() reads work correctly. The host validates state values before
    // engaging the motor.

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        while (true) { /* halt: STEP_PIN is not Timer1 OC1A/OC1B */ }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);
    stepper->setAutoEnable(false);

    int8_t rc_speed = stepper->setSpeedInUs(MOTOR_MIN_STEP_US);
    if (rc_speed != 0)
    {
        while (true) {}
    }
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    while (!Serial) { ; }
    last_sample_us = micros();
}

void loop()
{
    uint32_t now_us = micros();
    if ((uint32_t)(now_us - last_sample_us) >= SAMPLE_PERIOD_US)
    {
        last_sample_us = now_us;
        sampleState();
    }
    if (Serial.available() > 0)
    {
        handleCommand();
    }
    // Host-liveness watchdog: if the host went quiet mid-episode, stop
    // the motor instead of executing its last accel command forever.
    // Timestamp taken fresh here (not the `now_us` from the top of loop())
    // because handleCommand() above may have just refreshed last_cmd_us
    // via its own micros() call — comparing against the stale, earlier
    // now_us would make (now_us - last_cmd_us) underflow (unsigned) into
    // a huge value on the very tick that arms the watchdog, instantly
    // self-tripping before the motor ever gets to move.
    uint32_t watchdog_check_us = micros();
    if (motor_engaged && watchdog_armed &&
        (uint32_t)(watchdog_check_us - last_cmd_us) > WATCHDOG_TIMEOUT_US)
    {
        motor_engaged = false;
        watchdog_armed = false;
        stepper->forceStop();
        stepper->disableOutputs();
    }
}

/*
 * Append a (time, motor_position, pendulum_position) sample to the ring
 * buffer and update the pendulum wraparound accumulator.
 */
void sampleState()
{
    int32_t motor_step = stepper->getCurrentPosition();
    // Direct polling TWI read — immune to Timer1 ISR interference.
    // Masks Timer1 interrupts for the ~500 µs read window to avoid any
    // preemption of the busy-wait loops. The Wire ISR is also suspended
    // inside as5600_read_polling() via TWCR.TWIE.
    uint8_t saved_timsk1 = TIMSK1;
    TIMSK1 = 0;
    long raw = 0;
    bool i2c_ok = as5600_read_polling(&raw);
    TIMSK1 = saved_timsk1;

    // Pendulum wraparound tracking — only update on successful I²C reads.
    // A failed read (i2c_ok=false) keeps pen_raw_prev unchanged so the next
    // successful read computes a valid delta from the last known good position.
    // Feeding raw=0 (failure sentinel) into the accumulator when pen_raw_prev
    // is non-zero produces a spurious −14° step that corrupts every subsequent
    // observation.
    if (i2c_ok)
    {
        if (pen_raw_prev < 0)
        {
            pen_raw_prev = raw;
        }
        else
        {
            long delta = raw - pen_raw_prev;
            if (delta >  2048) delta -= 4096;
            if (delta < -2048) delta += 4096;
            if (delta > PEN_RAW_MAX_DELTA_LSB || delta < -PEN_RAW_MAX_DELTA_LSB)
            {
                pen_raw_prev = raw;
            }
            else
            {
                pen_position_rad += (float)delta * ((2.0f * PI) / 4096.0f);
                pen_raw_prev = raw;
            }
        }
    }

    motor_step_buf[buf_head] = motor_step;
    pen_rad_buf[buf_head]    = pen_position_rad;
    time_us_buf[buf_head]    = last_sample_us;
    buf_head = (buf_head + 1) % BUFFER_SIZE;
    if (buf_head == 0) buf_filled = true;
}

/*
 * Compute velocity over the most recent VEL_WINDOW samples as
 * (newest - oldest)/Δt. Returns 0 until the buffer holds enough samples.
 */
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s)
{
    uint8_t n_samples = buf_filled ? BUFFER_SIZE : buf_head;
    if (n_samples < VEL_WINDOW)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1)            % BUFFER_SIZE);
    uint8_t oldest = (uint8_t)((buf_head + BUFFER_SIZE - VEL_WINDOW)   % BUFFER_SIZE);

    uint32_t t_new = time_us_buf[newest];
    uint32_t t_old = time_us_buf[oldest];
    float dt_s = (float)((uint32_t)(t_new - t_old)) * 1e-6f;
    if (dt_s <= 0.0f)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    int32_t motor_step_delta = motor_step_buf[newest] - motor_step_buf[oldest];
    *motor_vel_rad_s = ((float)motor_step_delta * ((2.0f * PI) / 1600.0f)) / dt_s;

    float pen_delta = pen_rad_buf[newest] - pen_rad_buf[oldest];
    *pen_vel_rad_s = pen_delta / dt_s;
}

/*
 * Handle incoming commands.
 */
void handleCommand()
{
    uint8_t command = Serial.read();

    // Any received byte proves the host is alive — refresh the watchdog.
    last_cmd_us = micros();

    switch (command)
    {
    case CMD_READY:
        Serial.write(CMD_READY);
        break;

    case CMD_GET_STATE:
        sendState();
        break;

    case CMD_SET_ACCEL:
        {
            // Read 4 bytes (size of float) with a short timeout. If a byte
            // gets dropped — UART overrun under stepper EMI, or a flipped
            // command byte that desyncs the parser — readBytes returns
            // short and we bail. The next command re-syncs.
            Serial.setTimeout(5);
            float accel_rad_s2;
            size_t n = Serial.readBytes((char *)&accel_rad_s2, sizeof(float));
            Serial.setTimeout(1000);  // restore Stream default
            if (n != sizeof(float)) break;

            if (!motor_engaged) break;

            // Convert rad/s² to steps/s² (1600 microsteps per revolution).
            // moveByAcceleration takes int32_t.
            int32_t accel_steps_s2 =
                (int32_t)(accel_rad_s2 * (1600.0f / (2.0f * PI)));

            // Position-limit safety: past the rail, ignore the host's
            // command and actively brake instead. Just zeroing accel here
            // is dangerous — with allow_reverse=true, an accel of 0 means
            // "hold current speed", so a motor at +5 rad/s heading
            // outbound would coast indefinitely if the host went quiet.
            // Brake with a fixed opposing accel so we decelerate
            // regardless of host liveness.
            int32_t cur_pos = stepper->getCurrentPosition();
            if (cur_pos >= MOTOR_SAFE_LIMIT_STEPS)
            {
                accel_steps_s2 = -MOTOR_BRAKE_ACCEL_STEPS_S2;
            }
            else if (cur_pos <= -MOTOR_SAFE_LIMIT_STEPS)
            {
                accel_steps_s2 = +MOTOR_BRAKE_ACCEL_STEPS_S2;
            }

            // allow_reverse=true makes the library decelerate smoothly through
            // zero when the sign of accel opposes the current velocity — no
            // state machine needed on our side.
            stepper->moveByAcceleration(accel_steps_s2, true);

            // First accel command of this engagement: the control loop is
            // live, arm the watchdog.
            watchdog_armed = true;
        }
        break;

    case CMD_ENGAGE_MOTOR:
        motor_engaged = true;
        watchdog_armed = false;  // armed by the first CMD_SET_ACCEL
        stepper->enableOutputs();
        // Start in zero-accel state. moveByAcceleration(0, true) means
        // "hold current speed", and since we've just enabled the driver
        // the stepper is at rest — so this leaves it at rest until the
        // host sends its first CMD_SET_ACCEL.
        // Note: ENGAGE does not re-zero `getCurrentPosition()` or
        // `pen_position_rad`. Position counters persist across
        // engage/disengage cycles (and across episodes) by design — the
        // host is responsible for tracking the frame.
        stepper->moveByAcceleration(0, true);
        break;

    case CMD_DISENGAGE_MOTOR:
        motor_engaged = false;
        watchdog_armed = false;
        // forceStop() drains the Timer1 step queue immediately; without this,
        // queued steps would continue advancing the firmware position counter
        // even though the driver's enable pin is HIGH.
        stepper->forceStop();
        stepper->disableOutputs();
        break;

    case CMD_TARE_PENDULUM:
        // Re-zero pen_position_rad to the current AS5600 reading. Used by
        // real_env.reset() after the pendulum has settled at rest so that
        // each fine-tune episode samples a fresh bias from the rig's
        // static-friction-bounded rest distribution (±1.9°). Without this,
        // all fine-tune episodes share the firmware-boot bias and the
        // policy overfits to that single calibration offset.
        //
        // Shift pen_position_rad AND every entry in pen_rad_buf by the
        // current pen_position_rad. The buffer's relative deltas (used
        // by computeVelocities) are preserved, so velocity calculation
        // continues uninterrupted across the tare. Disable interrupts so
        // sampleState() doesn't run mid-update with a partially-shifted
        // buffer.
        noInterrupts();
        {
            float offset = pen_position_rad;
            for (uint8_t i = 0; i < BUFFER_SIZE; i++)
            {
                pen_rad_buf[i] -= offset;
            }
            pen_position_rad = 0.0f;
        }
        interrupts();
        Serial.write(CMD_TARE_PENDULUM);  // ack
        break;

    case CMD_DEBUG_I2C:
        {
            // Repeated-START read with diagnostics. Short delay ensures
            // ≥200 µs bus-free time after any preceding sampleState() STOP.
            // Returns 4 bytes: [endTransmission_rc, requestFrom_n, raw_hi, raw_lo]
            delayMicroseconds(200);
            uint8_t et_rc, rf_n;
            uint16_t raw16 = 0;
            uint8_t saved = TIMSK1;
            TIMSK1 &= ~(_BV(OCIE1A) | _BV(TOIE1));
            Wire.beginTransmission(0x36);
            Wire.write(0x0C);
            et_rc = (uint8_t)Wire.endTransmission(false);
            if (et_rc == 0)
            {
                rf_n = (uint8_t)Wire.requestFrom((uint8_t)0x36, (uint8_t)2);
                if (rf_n == 2)
                {
                    raw16 = ((uint16_t)Wire.read() << 8) | Wire.read();
                    raw16 &= 0x0FFF;
                }
            }
            else
            {
                rf_n = 0;
            }
            TIMSK1 = saved;
            Serial.write(et_rc);
            Serial.write(rf_n);
            Serial.write((uint8_t)(raw16 >> 8));
            Serial.write((uint8_t)(raw16 & 0xFF));
        }
        break;

    default:
        break;
    }
}

/*
 * Send the current state of the system:
 * - Current time in microseconds   (4 bytes, uint32)
 * - Stepper motor position in rad  (4 bytes, float)
 * - Pendulum position in rad       (4 bytes, float)
 * - Stepper motor velocity in rad/s (4 bytes, float)
 * - Pendulum velocity in rad/s     (4 bytes, float)
 */
void sendState()
{
    // Take the most recent buffer entry as the instantaneous position,
    // and compute velocity from the (newest - oldest)/Δt regression
    // window. The timestamp is the sample time of `newest` — not a
    // fresh micros() — so the host gets a self-consistent (t, pos, vel)
    // tuple it can time-align without inheriting up-to-one-sample of
    // bias.
    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1) % BUFFER_SIZE);
    uint32_t current_time = time_us_buf[newest];
    float motor_position_radians = stepsToRadians(motor_step_buf[newest]);
    float pendulum_position_radians = pen_rad_buf[newest];

    float motor_velocity_rad_s, pendulum_velocity_rad_s;
    computeVelocities(&motor_velocity_rad_s, &pendulum_velocity_rad_s);

    // Flip the signs of the motor and pendulum positions / velocities to
    // match the sim-frame convention the Python clients expect.
    motor_position_radians    *= -1;
    pendulum_position_radians *= -1;
    motor_velocity_rad_s      *= -1;
    pendulum_velocity_rad_s   *= -1;

    // Pack and send the data
    Serial.write((byte *)&current_time, sizeof(current_time));
    Serial.write((byte *)&motor_position_radians, sizeof(motor_position_radians));
    Serial.write((byte *)&pendulum_position_radians, sizeof(pendulum_position_radians));
    Serial.write((byte *)&motor_velocity_rad_s, sizeof(motor_velocity_rad_s));
    Serial.write((byte *)&pendulum_velocity_rad_s, sizeof(pendulum_velocity_rad_s));
}
