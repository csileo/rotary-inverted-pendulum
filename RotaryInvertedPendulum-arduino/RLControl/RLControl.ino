/**
 * RLControl.ino — Standalone on-device RL controller for the rotary inverted pendulum.
 *
 * Runs a distilled student MLP (5 -> 32 -> 32 -> 1, ReLU/ReLU/tanh, ~5 KB
 * float32 weights in PROGMEM) at a fixed 35 Hz to swing up + balance the
 * pendulum without any laptop tether. Distilled from
 * `runs/async_35hz_v2_extend/last.zip` via `distill.py` and exported by
 * `export_weights.py`.
 *
 * Step generation runs from a Timer1 ISR via FastAccelStepper. The main loop
 * is therefore free to spend ~15 ms on inference without stalling the stepper
 * acceleration ramp — earlier AccelStepper-based revisions had to interleave
 * stepper.run() calls inside the MAC loops to keep the motor responsive.
 *
 * Wiring: STEP must be on pin 9 (Timer1 OC1A on ATmega328); DIR on pin 2 and
 * ENABLE on pin 5 are unconstrained.
 *
 * Frame conventions (match `LowLevelServer` + `run_policy.py`):
 *   - The policy was trained with motor_pos and phi in the Arduino's raw
 *     stepper frame. LowLevelServer flips signs on get_state output and
 *     run_policy.py un-flips on receive — net no-op. So in this standalone
 *     sketch we use the raw frame directly: NO sign flip on read or write.
 *   - phi = 0 means pendulum hanging down (encoder zeros at boot).
 *   - theta = wrap_pi(phi - pi); theta = 0 means upright.
 *   - motor_pos = 0 at boot (stepper.currentPosition() starts at 0).
 *
 * Boot procedure:
 *   1. Power on or finish flashing. The sketch waits for a valid AS5600
 *      magnet detection, then waits a 1 s settle delay before engaging.
 *      The pose at the END of that delay becomes the policy's frame
 *      (phi = 0 = current pendulum angle, motor_pos = 0 = current
 *      stepper position). UX: have the pendulum hanging straight down
 *      before / during the 1 s delay; the LED is solid HIGH then drops
 *      to LOW once the motor engages.
 *   2. The policy runs; it swings up and balances.
 *   3. If the motor reaches a hard limit, the sketch disengages and waits.
 *      Press 'E' (after re-positioning) to re-engage with a fresh frame.
 *
 * Serial commands (500 kbaud, optional — the sketch is fully autonomous):
 *   'P' / 'p' : toggle CSV telemetry
 *   'E' / 'e' : engage motor (re-arm after a hard-limit trip)
 *   'D' / 'd' : disengage motor (manual stop)
 *   'M' / 'm' : print AS5600 magnet diagnostics
 *
 * Telemetry CSV (when toggled on, ~35 Hz):
 *   t_us, motor_pos_rad×1000, phi_rad×1000, action×1000, state, freq_hz, overruns
 */

#include <FastAccelStepper.h>
#include <AS5600.h>
#include <Wire.h>

// Define POLICY_QUANTISED to use the int8/QAT student exported by
// `export_weights_quantised.py`. Default (undefined) uses the float
// student exported by `export_weights.py`. The float build is the
// canonical production path; the quantised build is a Phase-5.5 stretch
// experiment — see docs/quantisation.md.
// #define POLICY_QUANTISED

#ifdef POLICY_QUANTISED
#include "policy_weights_quantised.h"
#else
#include "policy_weights.h"
#endif

// =============================================================================
// PINS
// =============================================================================
// On the ATmega328 (Nano), FastAccelStepper drives STEP from a Timer1
// hardware ISR — STEP must therefore be on pin 9 (OC1A) or pin 10 (OC1B).
// We use pin 9 by convention. DIR and ENABLE can be any digital pin.
const int DIR_PIN = 2;
const int STEP_PIN = 9;
const int ENABLE_PIN = 5;

// =============================================================================
// HARDWARE CONSTANTS
// =============================================================================
const long STEPS_PER_REVOLUTION = 200L * 8L;  // 200 full × 8 microsteps
const float STEPS_PER_RAD = STEPS_PER_REVOLUTION / (2.0f * (float)PI);
const float RAD_PER_STEP = (2.0f * (float)PI) / (float)STEPS_PER_REVOLUTION;

// =============================================================================
// COMMUNICATION
// =============================================================================
const long SERIAL_BAUD_RATE = 500000;  // matches PIDControl / SysIdRecord
const long I2C_CLOCK_HZ = 400000;

// =============================================================================
// MOTOR ENVELOPE.
//
// On a 16 MHz Nano with a *single* stepper, FastAccelStepper raises its
// internal max_speed_in_ticks to TICKS_PER_S/50000 inside
// `StepperQueue::adjustSpeedToStepperCount()` (pd_avr/avr_queue.cpp:328) —
// so up to 50 kSteps/s is permitted. (The fallback before adjustment is
// only 1 kStep/s, which is what initially caused setSpeedInHz to fail.)
//
// Acceleration A/B-tested 50 k vs 100 k on the rig: 100 k sounds buzzy
// and over-drives the policy's catch logic (multi-revolution spin
// instead of balance). 50 k gives a smooth whirr and matches the
// AccelStepper-era LowLevelServer setting — keep both sketches synced.
// =============================================================================
const uint32_t MOTOR_MAX_SPEED = 50000;
const int32_t MOTOR_ACCELERATION = 50000;

// =============================================================================
// CONTROL PARAMETERS
// =============================================================================
// Fixed control rate — MUST match the rate the policy was trained at.
// `runs/async_35hz_v2_extend` was trained at 35 Hz. CONTROL_PERIOD_US =
// round(1e6 / 35) = 28571.
const float CONTROL_FREQUENCY_HZ = 35.0f;
const unsigned long CONTROL_PERIOD_US = (unsigned long)(1000000.0f / CONTROL_FREQUENCY_HZ);
const float CONTROL_DT_S = 1.0f / CONTROL_FREQUENCY_HZ;

// Per-step action scale: matches `max_action_delta_rad=0.10` in pendulum_env.py
// and run_policy.py — the policy's [-1,+1] tanh output is scaled by this to
// produce the per-step motor target delta.
const float MAX_ACTION_DELTA_RAD = 0.10f;

// Velocity finite-diff low-pass cutoff — auto-derived in run_policy.py as
// min(20, max(10, 0.4 × control_freq)). At 35 Hz: 14 Hz.
//   alpha = dt / (RC + dt),  RC = 1 / (2π × 14 Hz) ≈ 0.01137 s
//   alpha = (1/35) / (0.01137 + 1/35) ≈ 0.715
// We compute it explicitly to avoid a magic number.
const float VEL_FILTER_CUTOFF_HZ = 14.0f;

// Motor position safety limits in policy frame.
//   SAFE_LIMIT (±125°) — matches MOTOR_SAFE_LIMIT_RAD in pendulum_env.py:45.
//     Used to clip the integrated motor target so the policy never *commands*
//     past the safe envelope.
//   HARD_LIMIT (±132°) — slightly inside the ±135° mechanical hard stops noted
//     in RL_PLAN.md. Crossing it disengages the motor and returns to WAITING.
const float MOTOR_SAFE_LIMIT_RAD = 2.18166f;   // 125° × π/180
const float MOTOR_HARD_LIMIT_RAD = 2.30383f;   // 132° × π/180

// =============================================================================
// STATE
// =============================================================================
// FastAccelStepper uses an engine+stepper-pointer pattern: the engine owns
// the Timer1 ISR and dispenses up to 3 stepper handles connected to specific
// hardware pins. We only need one stepper here.
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
AS5600 as5600;

enum State { WAITING, RUNNING };
State state = WAITING;

// Filtered velocities (rad/s) in policy frame.
float motor_vel_f = 0.0f;
float pen_vel_f = 0.0f;

// Previous-step positions for finite-diff velocity (policy frame).
float motor_pos_prev = 0.0f;
float phi_prev = 0.0f;

// Commanded motor target in policy frame (radians). Integrated from the
// policy's per-step action.
float motor_target_rad = 0.0f;

// Per-tick low-pass coefficient. Computed in setup() once, since CONTROL_DT_S
// and VEL_FILTER_CUTOFF_HZ are compile-time constants.
float vel_alpha = 0.0f;

// Telemetry / diagnostics
unsigned int loop_overruns = 0;
unsigned int loop_count_for_freq = 0;
unsigned long prev_time_us = 0;
bool print_enabled = false;
float last_action = 0.0f;

// =============================================================================
// UTILITY
// =============================================================================

static inline float wrap_pi(float x)
{
    // ((x + π) mod 2π) - π
    while (x >  (float)PI) x -= 2.0f * (float)PI;
    while (x < -(float)PI) x += 2.0f * (float)PI;
    return x;
}

static inline float read_motor_pos_rad()
{
    return (float)stepper->getCurrentPosition() * RAD_PER_STEP;
}

/**
 * Read the AS5600 with multi-revolution tracking; returns cumulative angle in
 * radians, zeroed by the most recent reset_pendulum_tracking() call (or boot
 * if never called). Same algorithm as LowLevelServer's
 * convertRawAngleToRadians() but with sign convention left at raw (we don't
 * apply the asymmetric sign flip the LowLevelServer applies on output, since
 * there's no client to un-flip).
 *
 * Re-zeroing lives here because the policy's frame requires `phi = 0` ↔
 * pendulum hanging down. If we captured the zero at boot — i.e. whatever
 * angle the pendulum was at the moment `arduino-cli upload` finished — the
 * policy would interpret the user's "hanging" position as some random
 * theta and command nonsensical actions. Instead we re-zero at every
 * (re-)engagement, after the user has had time to position the rig.
 */
static volatile bool _encoder_zero_pending = true;

static void reset_pendulum_tracking()
{
    _encoder_zero_pending = true;
}

static float read_pendulum_rad()
{
    const long AS5600_RES = 4096;
    const long WRAP_THRESH = AS5600_RES / 2;
    const float RAD_PER_SEG = (2.0f * (float)PI) / (float)AS5600_RES;

    static long raw_prev = 0;
    static float pos = 0.0f;

    long raw = (long)as5600.rawAngle();

    if (_encoder_zero_pending)
    {
        raw_prev = raw;
        pos = 0.0f;
        _encoder_zero_pending = false;
        return 0.0f;
    }

    long delta = raw - raw_prev;
    if (delta >  WRAP_THRESH) delta -= AS5600_RES;
    if (delta < -WRAP_THRESH) delta += AS5600_RES;

    pos += (float)delta * RAD_PER_SEG;
    raw_prev = raw;
    return pos;
}

// =============================================================================
// POLICY FORWARD PASS
// =============================================================================
//
// 5 -> H -> H -> 1 MLP, ReLU/ReLU/tanh. Weights live in PROGMEM and are
// read with pgm_read_*(); only the H+H activation buffers + the input
// live in SRAM.
//
// Step generation runs from a Timer1 ISR (FastAccelStepper), so the
// inference time has no effect on motor stepping — no interleaved
// stepper polling needed inside the MAC loops.
//
// Two implementations live behind a compile-time switch (POLICY_QUANTISED):
//
//   Float path (default):  ~12 µs/MAC software float, ~5 ms at H=16.
//   Int8 path (quantised): ~5 cycles/MAC int8 MUL, ~0.4 ms at H=16
//                          (~10× faster). See docs/quantisation.md.
//
// Both paths take the same (obs, action*) signature so the caller doesn't
// care which is compiled in.

#ifdef POLICY_QUANTISED

// -----------------------------------------------------------------------------
// Int8 forward pass — symmetric per-tensor quantisation.
//
// Per-layer:
//   accum_i32 = bias_i32 + sum( W_int8 * x_int8 )
//   For hidden layers: y_int8 = clamp((accum * M_q15) >> 15, 0, 127)
//                      (ReLU folds in here as the lower clamp).
//   For the final layer: y_float = accum_i32 * dequant_l3, then tanh.
//
// The single-int32 multiply for the rescale (accum_i32 * M_q15) doesn't
// overflow on an ATmega328 because for typical scales accum is at most
// ~2^18 and M_q15 fits in int16, so the product fits in int32. The
// export script raises if either bound is violated.

static void policy_forward(const float obs[POLICY_OBS_DIM], float* action)
{
    int8_t x[POLICY_OBS_DIM];
    int8_t h1[POLICY_HIDDEN_DIM];
    int8_t h2[POLICY_HIDDEN_DIM];

    // Per-channel input quantisation: each obs dim has its own inverse-scale
    // factor. (Per-channel input scales recover precision near the
    // equilibrium where motor_pos / sin / cos are small.)
    for (int j = 0; j < POLICY_OBS_DIM; j++)
    {
        float inv_s = pgm_read_float(&POLICY_INV_SCALE_OBS_IN[j]);
        float q = obs[j] * inv_s;
        long qi = (long)(q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        x[j] = (int8_t)qi;
    }

    // Layer 1: int8 matmul + bias + per-row Q15 rescale + ReLU.
    // M_Q15_L1[i] is per output neuron — each row gets its own rescale
    // factor, which has the per-channel input scales already absorbed.
    for (int i = 0; i < POLICY_HIDDEN_DIM; i++)
    {
        int32_t accum = (int32_t)pgm_read_dword(&POLICY_B1[i]);
        for (int j = 0; j < POLICY_OBS_DIM; j++)
        {
            int8_t w = (int8_t)pgm_read_byte(&POLICY_W1[i][j]);
            accum += (int32_t)w * (int32_t)x[j];
        }
        int16_t m_q15 = (int16_t)pgm_read_word(&POLICY_M_Q15_L1[i]);
        int32_t scaled = (accum * (int32_t)m_q15 + (1L << 14)) >> 15;
        if (scaled > 127) scaled = 127;
        if (scaled < 0)   scaled = 0;   // ReLU
        h1[i] = (int8_t)scaled;
    }

    // Layer 2: same shape, per-row rescale.
    for (int i = 0; i < POLICY_HIDDEN_DIM; i++)
    {
        int32_t accum = (int32_t)pgm_read_dword(&POLICY_B2[i]);
        for (int j = 0; j < POLICY_HIDDEN_DIM; j++)
        {
            int8_t w = (int8_t)pgm_read_byte(&POLICY_W2[i][j]);
            accum += (int32_t)w * (int32_t)h1[j];
        }
        int16_t m_q15 = (int16_t)pgm_read_word(&POLICY_M_Q15_L2[i]);
        int32_t scaled = (accum * (int32_t)m_q15 + (1L << 14)) >> 15;
        if (scaled > 127) scaled = 127;
        if (scaled < 0)   scaled = 0;
        h2[i] = (int8_t)scaled;
    }

    // Layer 3: int8 matmul + bias, per-output dequantise to float, then tanh.
    int32_t accum = (int32_t)pgm_read_dword(&POLICY_B3[0]);
    for (int j = 0; j < POLICY_HIDDEN_DIM; j++)
    {
        int8_t w = (int8_t)pgm_read_byte(&POLICY_W3[0][j]);
        accum += (int32_t)w * (int32_t)h2[j];
    }
    float dequant = pgm_read_float(&POLICY_DEQUANT_L3[0]);
    float y = (float)accum * dequant;
    *action = tanhf(y);
}

#else  // POLICY_QUANTISED — float path below

// -----------------------------------------------------------------------------
// Float forward pass — production default.

static void policy_forward(const float obs[POLICY_OBS_DIM], float* action)
{
    float h1[POLICY_HIDDEN_DIM];
    float h2[POLICY_HIDDEN_DIM];

    // Layer 1: obs (5) -> h1 (H), ReLU.
    for (int i = 0; i < POLICY_HIDDEN_DIM; i++)
    {
        float sum = pgm_read_float(&POLICY_B1[i]);
        for (int j = 0; j < POLICY_OBS_DIM; j++)
        {
            sum += obs[j] * pgm_read_float(&POLICY_W1[i][j]);
        }
        h1[i] = sum > 0.0f ? sum : 0.0f;
    }

    // Layer 2: h1 (H) -> h2 (H), ReLU.
    for (int i = 0; i < POLICY_HIDDEN_DIM; i++)
    {
        float sum = pgm_read_float(&POLICY_B2[i]);
        for (int j = 0; j < POLICY_HIDDEN_DIM; j++)
        {
            sum += h1[j] * pgm_read_float(&POLICY_W2[i][j]);
        }
        h2[i] = sum > 0.0f ? sum : 0.0f;
    }

    // Layer 3: h2 (H) -> action (1), tanh.
    float sum = pgm_read_float(&POLICY_B3[0]);
    for (int j = 0; j < POLICY_HIDDEN_DIM; j++)
    {
        sum += h2[j] * pgm_read_float(&POLICY_W3[0][j]);
    }
    *action = tanhf(sum);
}

#endif  // POLICY_QUANTISED

// =============================================================================
// STATE MACHINE
// =============================================================================

static void prime_initial_state()
{
    // Mirror run_policy.py:131-134 priming: target = current motor pos, zero
    // velocities, zero filters, so the first finite-diff reads as 0.
    float motor_pos = read_motor_pos_rad();
    float phi = read_pendulum_rad();  // returns 0 right after reset_pendulum_tracking()
    motor_target_rad = constrain(motor_pos, -MOTOR_SAFE_LIMIT_RAD, MOTOR_SAFE_LIMIT_RAD);
    motor_pos_prev = motor_pos;
    phi_prev = phi;
    motor_vel_f = 0.0f;
    pen_vel_f = 0.0f;
}

static void transition_to_running()
{
    // Recapture both the encoder zero (phi=0 ↔ current pendulum position)
    // and the stepper origin (motor_pos=0 ↔ current motor position) so the
    // policy sees the same frame conventions it was trained in regardless
    // of how the user reset/positioned the rig.
    reset_pendulum_tracking();
    stepper->setCurrentPosition(0);
    prime_initial_state();
    stepper->enableOutputs();
    state = RUNNING;
}

static void transition_to_waiting()
{
    stepper->forceStop();
    stepper->disableOutputs();
    state = WAITING;
}

// =============================================================================
// CONTROL TICK (called once per CONTROL_PERIOD_US)
// =============================================================================

static void control_tick(float dt_s)
{
    // 1. Read state.
    float motor_pos = read_motor_pos_rad();
    float phi = read_pendulum_rad();

    // 2. Hard-limit safety: trip back to WAITING if motor strayed past the
    // mechanical envelope. The policy is supposed to keep us inside SAFE_LIMIT,
    // but trust nothing; AccelStepper might still be ramping past a recently
    // updated target.
    if (fabs(motor_pos) > MOTOR_HARD_LIMIT_RAD)
    {
        transition_to_waiting();
        return;
    }

    // 3. Finite-diff + IIR low-pass velocities. Use the *measured* dt
    // (passed in from the loop) rather than the nominal CONTROL_DT_S so a
    // long-running tick doesn't inflate the velocity estimate. Matches
    // run_policy.py's dt_meas approach. The IIR alpha is left at its
    // CONTROL_DT_S-derived value because typical dt jitter is <5 % and
    // recomputing alpha each tick costs more than it pays back.
    float motor_vel_inst = (motor_pos - motor_pos_prev) / dt_s;
    float pen_vel_inst = (phi - phi_prev) / dt_s;
    motor_vel_f += vel_alpha * (motor_vel_inst - motor_vel_f);
    pen_vel_f += vel_alpha * (pen_vel_inst - pen_vel_f);
    motor_pos_prev = motor_pos;
    phi_prev = phi;

    // 4. Build observation: [motor_pos, sin(theta), cos(theta), motor_vel, pen_vel].
    float theta = wrap_pi(phi - (float)PI);
    float obs[POLICY_OBS_DIM];
    obs[0] = motor_pos;
    obs[1] = sinf(theta);
    obs[2] = cosf(theta);
    obs[3] = motor_vel_f;
    obs[4] = pen_vel_f;

    // 5. Forward pass.
    float action;
    policy_forward(obs, &action);
    if (action > 1.0f) action = 1.0f;
    else if (action < -1.0f) action = -1.0f;
    last_action = action;

    // 6. Integrate into motor target (clipped to safe envelope) and command.
    motor_target_rad += action * MAX_ACTION_DELTA_RAD;
    if (motor_target_rad >  MOTOR_SAFE_LIMIT_RAD) motor_target_rad =  MOTOR_SAFE_LIMIT_RAD;
    if (motor_target_rad < -MOTOR_SAFE_LIMIT_RAD) motor_target_rad = -MOTOR_SAFE_LIMIT_RAD;
    int32_t target_steps = (int32_t)(motor_target_rad * STEPS_PER_RAD);
    stepper->moveTo(target_steps);
}

// =============================================================================
// SERIAL
// =============================================================================

static void handle_serial()
{
    if (!Serial.available()) return;
    char cmd = Serial.read();
    while (Serial.available()) Serial.read();
    switch (cmd)
    {
    case 'P': case 'p': print_enabled = !print_enabled; break;
    case 'E': case 'e': if (state == WAITING) transition_to_running(); break;
    case 'D': case 'd': if (state == RUNNING) transition_to_waiting(); break;
    case 'M': case 'm':
        Serial.print(F("[AS5600] magnet="));
        if (as5600.magnetTooWeak()) Serial.println(F("WEAK"));
        else if (as5600.magnetTooStrong()) Serial.println(F("STRONG"));
        else Serial.println(F("OK"));
        break;
    }
}

static void print_telemetry(unsigned long now_us, unsigned int freq_hz)
{
    if (!print_enabled) return;
    // CSV: t_us, motor_pos_rad*1000, phi_rad*1000, action*1000, state, freq_hz, overruns
    // Integer transmission avoids the ~500 µs Serial.print(float) cost.
    char buf[80];
    char* p = buf;
    ltoa((long)now_us, p, 10); p += strlen(p); *p++ = ',';
    ltoa((long)(read_motor_pos_rad() * 1000.0f), p, 10); p += strlen(p); *p++ = ',';
    ltoa((long)(read_pendulum_rad() * 1000.0f), p, 10); p += strlen(p); *p++ = ',';
    ltoa((long)(last_action * 1000.0f), p, 10); p += strlen(p); *p++ = ',';
    *p++ = (state == RUNNING) ? '1' : '0'; *p++ = ',';
    utoa(freq_hz, p, 10); p += strlen(p); *p++ = ',';
    utoa(loop_overruns, p, 10); p += strlen(p);
    *p = '\0';
    Serial.println(buf);
}

// =============================================================================
// LED
// =============================================================================

static void update_led()
{
    static unsigned long last_ms = 0;
    static bool on = false;
    unsigned long now = millis();
    unsigned long period = (state == RUNNING) ? 100 : 500;
    if (now - last_ms >= period)
    {
        last_ms = now;
        on = !on;
        digitalWrite(LED_BUILTIN, on ? HIGH : LOW);
    }
}

// =============================================================================
// SETUP / LOOP
// =============================================================================

void setup()
{
    Serial.begin(SERIAL_BAUD_RATE);
    Wire.begin();
    Wire.setClock(I2C_CLOCK_HZ);
    as5600.begin();

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Initialise FastAccelStepper. stepperConnectToPin must be on a Timer1
    // OC pin (pin 9 = OC1A on ATmega328); it returns NULL if the pin is
    // unsupported, which would silently disable stepping — guard with a
    // halt + LED-on so the failure is visible.
    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        Serial.println(F("[FATAL] FastAccelStepper failed to claim STEP pin"));
        digitalWrite(LED_BUILTIN, HIGH);
        while (true) {}
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);  // default low_active=true matches DRV8825
    stepper->setAutoEnable(false);      // we manually enable/disable on state changes
    int8_t rc_speed = stepper->setSpeedInHz(MOTOR_MAX_SPEED);
    int8_t rc_accel = stepper->setAcceleration(MOTOR_ACCELERATION);
    if (rc_speed != 0 || rc_accel != 0)
    {
        // Silent rejections will leave the stepper unable to issue any pulses.
        // Print a diagnostic and halt rather than booting into a dead-motor mode.
        Serial.print(F("[FATAL] FastAccelStepper config rejected: speed_rc="));
        Serial.print(rc_speed);
        Serial.print(F(" accel_rc="));
        Serial.println(rc_accel);
        digitalWrite(LED_BUILTIN, HIGH);
        while (true) {}
    }
    stepper->disableOutputs();

    // Compute IIR alpha from cutoff. Same form as run_policy.py.
    {
        float rc = 1.0f / (2.0f * (float)PI * VEL_FILTER_CUTOFF_HZ);
        vel_alpha = CONTROL_DT_S / (rc + CONTROL_DT_S);
    }

    while (!as5600.detectMagnet())
    {
        delay(500);
    }

    // Encoder zero is captured at engage time, not here — see
    // reset_pendulum_tracking() / transition_to_running().

    // Forward-pass self-test: compute the action for a fixed reference obs
    // and print it. Compare against the PyTorch student's prediction for
    // the same obs to confirm PROGMEM access + indexing are correct.
    // Re-derive expected values from the .pt file with the helper in
    // docs/end_to_end_runbook.md (step 6) — values are policy-specific
    // and change every distill.
    {
        float test_obs[POLICY_OBS_DIM];
        float test_act;
        // Hanging-down, still: [motor=0, sin(±π)=0, cos(±π)=-1, mvel=0, pvel=0]
        test_obs[0] = 0.0f; test_obs[1] = 0.0f; test_obs[2] = -1.0f;
        test_obs[3] = 0.0f; test_obs[4] = 0.0f;
        policy_forward(test_obs, &test_act);
        Serial.print(F("[boot] policy(hanging) = "));
        Serial.println(test_act, 6);
        // Upright, still: [motor=0, sin(0)=0, cos(0)=1, mvel=0, pvel=0]
        test_obs[2] = 1.0f;
        policy_forward(test_obs, &test_act);
        Serial.print(F("[boot] policy(upright) = "));
        Serial.println(test_act, 6);
    }

    // 1 s settle delay before engaging — gives the user a moment to verify
    // the pendulum is hanging straight down (LED stays HIGH during the
    // delay). Whatever pose the rig is in at the END of this delay
    // becomes the policy's frame (encoder zero + stepper origin captured
    // by transition_to_running).
    delay(1000);
    digitalWrite(LED_BUILTIN, LOW);
    transition_to_running();

    prev_time_us = micros();
}

void loop()
{
    // FastAccelStepper drives stepping from a Timer1 ISR — the main loop no
    // longer needs to call stepper.run() at all. The loop just paces control
    // ticks at the configured rate.
    unsigned long now_us = micros();
    unsigned long elapsed_us = now_us - prev_time_us;
    if (elapsed_us < CONTROL_PERIOD_US)
    {
        return;
    }

    if (elapsed_us > CONTROL_PERIOD_US * 3UL / 2UL)
    {
        loop_overruns++;
    }

    prev_time_us = now_us;
    loop_count_for_freq++;

    handle_serial();
    update_led();

    if (state == RUNNING)
    {
        control_tick(elapsed_us * 1e-6f);
    }

    // Telemetry every ~1 s (just print one line per second to avoid serial
    // overhead at 35 Hz). Each tick we already paid for one micros() call.
    static unsigned long last_print_us = 0;
    if (now_us - last_print_us >= 1000000UL)
    {
        unsigned int hz = (unsigned int)((unsigned long)loop_count_for_freq * 1000000UL
                                         / (now_us - last_print_us));
        print_telemetry(now_us, hz);
        loop_count_for_freq = 0;
        last_print_us = now_us;
    }
}
