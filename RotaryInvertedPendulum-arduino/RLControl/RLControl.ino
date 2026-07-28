/**
 * RLControl.ino — Standalone on-device RL controller for the rotary inverted pendulum.
 *
 * Runs a distilled student MLP at a fixed 35 Hz to swing up + balance the
 * pendulum without any laptop tether. Distilled from
 * `models/policy_working_balance.zip` (+ its real-rig replay buffer) via
 * `distill.py` and exported by `export_weights.py` — see
 * `src/rl/models/distill_working_balance_h32/README.md`.
 *
 * Step generation runs from a Timer1 ISR via FastAccelStepper. The main loop
 * is therefore free to spend time on inference without stalling stepping.
 *
 * Observation / action contract (MUST match pendulum_env.py — a mismatch
 * either desyncs silently or saturates the policy output):
 *   - Raw per-tick observation (6-dim, `pendulum_env.py:_obs()`):
 *     [motor_pos, sin(theta), cos(theta), motor_vel, pen_vel, prev_action].
 *   - The policy was trained with `--frame-stack 3` (see frame_stack.py):
 *     the last 3 raw observations, concatenated oldest -> newest, form the
 *     18-dim input the net actually sees. `POLICY_OBS_DIM` (from
 *     policy_weights.h) MUST equal RAW_OBS_DIM * FRAME_STACK — enforced by
 *     a static_assert below.
 *   - Action is accel-mode (`pendulum_env.py:step()`, "Mirrors
 *     FastAccelStepper's moveByAcceleration() behaviour"): the policy's
 *     [-1,+1] tanh output × MAX_ACCEL_RAD_S2 is commanded straight to
 *     `stepper->moveByAcceleration()` every tick — same as LowLevelServer's
 *     CMD_SET_ACCEL handler. No position-delta integration on our side; the
 *     FastAccelStepper ISR does the accel -> velocity -> position work.
 *   - motor_vel / pen_vel are computed from a ring-buffer window
 *     ((newest - oldest)/dt over VEL_WINDOW samples at SAMPLE_PERIOD_US),
 *     identical to LowLevelServer's computeVelocities() — this is the
 *     estimator the real-rig replay data (used to distill this policy) was
 *     actually generated with, per frame_stack.py's docstring.
 *
 * Frame conventions (match `LowLevelServer` + `run_policy.py`):
 *   - The policy was trained with motor_pos and phi in the Arduino's raw
 *     stepper frame. LowLevelServer flips signs on get_state output and
 *     run_policy.py un-flips on receive — net no-op. So in this standalone
 *     sketch we use the raw frame directly: NO sign flip on read or write
 *     (same for the accel command: LowLevelServer's CMD_SET_ACCEL doesn't
 *     flip its input either).
 *   - phi = 0 means pendulum hanging down (encoder zeros at (re-)engage).
 *   - theta = wrap_pi(phi - pi); theta = 0 means upright.
 *   - motor_pos = 0 at (re-)engage (stepper->setCurrentPosition(0)).
 *
 * Safety (mirrors LowLevelServer's brake-at-rail, added after the
 * 2026-07-05 incident where an unbounded moveByAcceleration() wound the
 * motor into the hard stop and fried a Nano's USB-serial chip):
 *   - Past MOTOR_SAFE_LIMIT_RAD (±125°), the commanded accel is
 *     unconditionally overridden with full braking authority back toward
 *     center, regardless of what the policy asked for.
 *   - Past MOTOR_HARD_LIMIT_RAD (±132°, inside the ~135° mechanical hard
 *     stop), the sketch disengages outright and returns to WAITING as a
 *     second line of defense.
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
 *   3. If the motor reaches the hard limit, the sketch disengages and
 *      waits. Press 'E' (after re-positioning) to re-engage with a fresh
 *      frame.
 *
 * Serial commands (500 kbaud, optional — the sketch is fully autonomous):
 *   'P' / 'p' : toggle CSV telemetry
 *   'E' / 'e' : engage motor (re-arm after a hard-limit trip)
 *   'D' / 'd' : disengage motor (manual stop)
 *   'M' / 'm' : print AS5600 magnet diagnostics
 *
 * Telemetry CSV (when toggled on, ~1 Hz):
 *   t_us, motor_pos_rad×1000, phi_rad×1000, action×1000, state, freq_hz, overruns
 */

#include <FastAccelStepper.h>
#include <AS5600.h>
#include <Wire.h>

// Define exactly one of POLICY_QUANTISED_INT8 / POLICY_QUANTISED_INT16 to
// use a QAT student exported by `export_weights_quantised.py`; leave both
// undefined for the float student exported by `export_weights.py`.
//
// Float is out at H=32/18-in (~1632 MAC): the forward pass measures
// ~19.5 ms (extrapolated from the ~14.5 ms/1216-MAC figure in
// docs/quantisation.md), which eats most of the 28.57 ms / 35 Hz tick
// budget and was observed to knock the real control rate down to
// ~32-33 Hz — enough of a systematic (non-jitter) rate deficit to prevent
// a clean catch at the top of the swing-up.
//
// int8 (~0.4 ms/inference) fixed the rate deficit and produced a stable
// catch, but still falls occasionally on long runs — QAT val_mse=0.011192
// vs the float student's 0.0072, i.e. genuine int8 rounding error, not a
// rate problem (confirmed: the float student run tethered via
// run_policy.py holds upright=0.988 over 60s with the same weights).
//
// int16/int14 (~1.1 ms/inference, ~256x finer quantisation grid than int8
// per docs/quantisation.md's designated fallback) was tried and parked:
// layer 1's "absorb the per-channel input scale into the weights" trick
// (see export_weights_quantised.py) makes that layer's raw accumulator
// grow large enough at bits>=14 to starve the rescale multiplier of
// precision regardless of the Q-shift chosen — a structural mismatch
// between this architecture and wide bit-widths, not a bug to retry.
// int8 is current: export_weights_quantised.py's per-layer *adaptive*
// Q-shift (2026-07-28, replacing the old hardcoded Q15) already recovered
// most of int8's own headroom-vs-precision tradeoff — parity mean dropped
// from 1.8 LSB to 0.1 LSB with the exact same bit-width, at no extra cost.
#define POLICY_QUANTISED_INT8
// #define POLICY_QUANTISED_INT16

#if defined(POLICY_QUANTISED_INT8) && defined(POLICY_QUANTISED_INT16)
#error "define at most one of POLICY_QUANTISED_INT8 / POLICY_QUANTISED_INT16"
#endif

#if defined(POLICY_QUANTISED_INT8)
#include "policy_weights_quantised.h"
static_assert(POLICY_WEIGHT_BITS == 8,
              "policy_weights_quantised.h wasn't exported with --bits 8");
#elif defined(POLICY_QUANTISED_INT16)
#include "policy_weights_quantised_int16.h"
static_assert(POLICY_WEIGHT_BITS == 16,
              "policy_weights_quantised_int16.h wasn't exported with --bits 16");
#else
#include "policy_weights.h"
#endif

// =============================================================================
// OBSERVATION / FRAME STACKING
// =============================================================================
const int RAW_OBS_DIM = 6;   // [motor_pos, sin(theta), cos(theta), motor_vel, pen_vel, prev_action]
const int FRAME_STACK = POLICY_OBS_DIM / RAW_OBS_DIM;
static_assert(POLICY_OBS_DIM == RAW_OBS_DIM * FRAME_STACK,
              "policy_weights.h POLICY_OBS_DIM isn't a multiple of RAW_OBS_DIM=6 "
              "-- weights/sketch obs contract mismatch, regenerate weights or fix RAW_OBS_DIM");

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
// MOTOR ENVELOPE
// =============================================================================
// Velocity ceiling for moveByAcceleration(), matches LowLevelServer's
// MOTOR_MIN_STEP_US: MAX_VELOCITY_RAD_S=5 rad/s in pendulum_env.py, with
// headroom (~7 rad/s) so the firmware never clips before the sim-side
// velocity cap would.
const uint32_t MOTOR_MIN_STEP_US = 550;

// =============================================================================
// CONTROL PARAMETERS
// =============================================================================
// Fixed control rate — MUST match the rate the policy was trained at.
// CONTROL_PERIOD_US = round(1e6 / 35) = 28571.
const float CONTROL_FREQUENCY_HZ = 35.0f;
const unsigned long CONTROL_PERIOD_US = (unsigned long)(1000000.0f / CONTROL_FREQUENCY_HZ);

// Accel-mode action scale: matches MAX_ACCEL_RAD_S2 in pendulum_env.py and
// the brake authority in LowLevelServer.ino's CMD_SET_ACCEL handler.
const float MAX_ACCEL_RAD_S2 = 150.0f;

// Position sampling ring buffer (mirrors LowLevelServer.ino exactly — this
// is the estimator the real-rig replay data used to distill this policy
// was generated with; see frame_stack.py's docstring).
const uint16_t SAMPLE_PERIOD_US = 2000;
const uint8_t  BUFFER_SIZE      = 16;
const uint8_t  VEL_WINDOW       = 5;
const long     PEN_RAW_MAX_DELTA_LSB = 500;  // reject I2C-glitch wraps, see LowLevelServer.ino

// Motor position safety limits in policy frame.
//   SAFE_LIMIT (±125°) — matches MOTOR_SAFE_LIMIT_RAD in pendulum_env.py:45
//     and LowLevelServer.ino's MOTOR_SAFE_LIMIT_STEPS. Past it, the
//     commanded accel is unconditionally overridden with full braking
//     authority back toward center (see control_tick()).
//   HARD_LIMIT (±132°) — slightly inside the ±135° mechanical hard stops
//     (RL_PLAN.md). Crossing it disengages the motor and returns to
//     WAITING — a second line of defense behind the brake above.
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

// Ring buffer of (motor_step, pen_position_rad, t_us) samples, filled at
// SAMPLE_PERIOD_US independent of control-tick pacing — see sampleState().
static int32_t  motor_step_buf[BUFFER_SIZE];
static float    pen_rad_buf[BUFFER_SIZE];
static uint32_t time_us_buf[BUFFER_SIZE];
static uint8_t  buf_head = 0;
static bool     buf_filled = false;
static unsigned long last_sample_us = 0;

// Continuously-tracked pendulum angle (multi-revolution, AS5600 raw-LSB
// wraparound handled in sampleState()). Re-baselined at every (re-)engage.
static long  pen_raw_prev = -1;   // -1 = "next sample is the baseline"
static float pen_position_rad = 0.0f;

// Last action issued (feeds obs[5]=prev_action next tick — restores Markov
// property under action delay, matches run_policy.py / pendulum_env.py).
static float prev_action = 0.0f;

// Frame-stack buffer: oldest -> newest, RAW_OBS_DIM floats per frame.
static float frame_buf[FRAME_STACK][RAW_OBS_DIM];

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

// =============================================================================
// POSITION SAMPLING (2 kHz, independent of the 35 Hz control tick)
// =============================================================================
//
// Mirrors LowLevelServer.ino's sampleState()/computeVelocities() exactly —
// the training data's velocity estimator, per frame_stack.py's docstring.

static void sampleState()
{
    int32_t motor_step = stepper->getCurrentPosition();
    long raw = (long)as5600.rawAngle();

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
            // Almost certainly an I2C glitch, not real motion — resync the
            // baseline instead of integrating a spurious jump.
            pen_raw_prev = raw;
        }
        else
        {
            pen_position_rad += (float)delta * ((2.0f * (float)PI) / 4096.0f);
            pen_raw_prev = raw;
        }
    }

    motor_step_buf[buf_head] = motor_step;
    pen_rad_buf[buf_head]    = pen_position_rad;
    time_us_buf[buf_head]    = last_sample_us;
    buf_head = (buf_head + 1) % BUFFER_SIZE;
    if (buf_head == 0) buf_filled = true;
}

static void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s)
{
    uint8_t n_samples = buf_filled ? BUFFER_SIZE : buf_head;
    if (n_samples < VEL_WINDOW)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1)          % BUFFER_SIZE);
    uint8_t oldest = (uint8_t)((buf_head + BUFFER_SIZE - VEL_WINDOW) % BUFFER_SIZE);

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
    *motor_vel_rad_s = ((float)motor_step_delta * RAD_PER_STEP) / dt_s;

    float pen_delta = pen_rad_buf[newest] - pen_rad_buf[oldest];
    *pen_vel_rad_s = pen_delta / dt_s;
}

static void reset_pendulum_tracking()
{
    pen_raw_prev = -1;
    pen_position_rad = 0.0f;
    buf_head = 0;
    buf_filled = false;
}

// =============================================================================
// FRAME STACKING
// =============================================================================
// Matches frame_stack.py's FrameStacker: reset() fills all FRAME_STACK
// slots with the first frame (no fabricated history), push() shifts the
// window. Concatenation order is oldest -> newest.

static void frame_stacker_reset(const float frame[RAW_OBS_DIM])
{
    for (int f = 0; f < FRAME_STACK; f++)
        for (int j = 0; j < RAW_OBS_DIM; j++)
            frame_buf[f][j] = frame[j];
}

static void frame_stacker_push(const float frame[RAW_OBS_DIM])
{
    for (int f = 0; f < FRAME_STACK - 1; f++)
        for (int j = 0; j < RAW_OBS_DIM; j++)
            frame_buf[f][j] = frame_buf[f + 1][j];
    for (int j = 0; j < RAW_OBS_DIM; j++)
        frame_buf[FRAME_STACK - 1][j] = frame[j];
}

static void frame_stacker_get(float out[POLICY_OBS_DIM])
{
    for (int f = 0; f < FRAME_STACK; f++)
        for (int j = 0; j < RAW_OBS_DIM; j++)
            out[f * RAW_OBS_DIM + j] = frame_buf[f][j];
}

// =============================================================================
// POLICY FORWARD PASS
// =============================================================================
//
// H1 -> H2 -> 1 MLP, ReLU/ReLU/tanh. Weights live in PROGMEM and are
// read with pgm_read_*(); only the H+H activation buffers + the input
// live in SRAM.
//
// Step generation runs from a Timer1 ISR (FastAccelStepper), so the
// inference time has no effect on motor stepping — no interleaved
// stepper polling needed inside the MAC loops.
//
// Three implementations live behind the POLICY_QUANTISED_INT8 /
// POLICY_QUANTISED_INT16 switch at the top of this file:
//
//   Float  (default):  ~190 cycles/MAC software float, ~14.5 ms/1216-MAC.
//   Int8   (quantised): ~5 cycles/MAC, ~0.4 ms/1216-MAC (~36x faster).
//   Int16  (quantised): ~15 cycles/MAC (4x 8x8 MUL, no native 16x16
//                        multiplier on AVR), ~1.1 ms/1216-MAC (~13x
//                        faster than float). See docs/quantisation.md.
//
// All three take the same (obs, action*) signature so the caller doesn't
// care which is compiled in.

#if defined(POLICY_QUANTISED_INT8)

// -----------------------------------------------------------------------------
// Int8 forward pass — symmetric per-tensor quantisation.
//
// Per-layer:
//   accum_i32 = bias_i32 + sum( W_int8 * x_int8 )
//   For hidden layers: y_int8 = clamp((accum * M_l1/l2) >> shift_l1/l2, 0, 127)
//                      (ReLU folds in here as the lower clamp).
//   For the final layer: y_float = accum_i32 * dequant_l3, then tanh.
//
// shift_l1/shift_l2 are PER-LAYER (not hardcoded to 15) — chosen at export
// time by calibrate_rescale() in export_weights_quantised.py against real
// on-distribution data, trading off rescale precision against int32
// overflow margin. See that script's module docstring for why a fixed
// Q15 rescale (the pre-2026-07-28 scheme) isn't safe to reuse verbatim at
// other bit-widths.

static void policy_forward(const float obs[POLICY_OBS_DIM], float* action)
{
    int8_t x[POLICY_OBS_DIM];
    int8_t h1[POLICY_HIDDEN_DIM];
    int8_t h2[POLICY_HIDDEN_DIM];

    const int32_t round_l1 = (POLICY_RESCALE_SHIFT_L1 > 0)
        ? (1L << (POLICY_RESCALE_SHIFT_L1 - 1)) : 0L;
    const int32_t round_l2 = (POLICY_RESCALE_SHIFT_L2 > 0)
        ? (1L << (POLICY_RESCALE_SHIFT_L2 - 1)) : 0L;

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

    // Layer 1: int8 matmul + bias + per-row rescale + ReLU.
    // POLICY_RESCALE_M_L1[i] is per output neuron — each row gets its own
    // rescale multiplier, which has the per-channel input scales already
    // absorbed.
    for (int i = 0; i < POLICY_HIDDEN_DIM; i++)
    {
        int32_t accum = (int32_t)pgm_read_dword(&POLICY_B1[i]);
        for (int j = 0; j < POLICY_OBS_DIM; j++)
        {
            int8_t w = (int8_t)pgm_read_byte(&POLICY_W1[i][j]);
            accum += (int32_t)w * (int32_t)x[j];
        }
        int16_t m = (int16_t)pgm_read_word(&POLICY_RESCALE_M_L1[i]);
        int32_t scaled = (accum * (int32_t)m + round_l1) >> POLICY_RESCALE_SHIFT_L1;
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
        int16_t m = (int16_t)pgm_read_word(&POLICY_RESCALE_M_L2[i]);
        int32_t scaled = (accum * (int32_t)m + round_l2) >> POLICY_RESCALE_SHIFT_L2;
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

#elif defined(POLICY_QUANTISED_INT16)

// -----------------------------------------------------------------------------
// Int16-storage forward pass — same structure as int8, wider grid (logical
// range is POLICY_WEIGHT_BITS, which may be < 16 — see distill_quantised.py's
// --bits docstring for why; the C storage type is int16_t either way).
//
// Per-layer:
//   accum_i32 = bias_i32 + sum( W_int16 * x_int16 )
//   For hidden layers: y = clamp((accum * M_l1/l2) >> shift_l1/l2, 0, max_int)
//   For the final layer: y_float = accum_i32 * dequant_l3, then tanh.
//
// The accumulator here is NOT automatically safe the way int8's is: a
// single 16-bit x 16-bit term can reach ~1e9, so summing
// POLICY_OBS_DIM/HIDDEN_DIM of them could in principle overflow int32
// (~2.15e9). export_weights_quantised.py's calibrate_rescale() checks this
// empirically against real data before ever writing this header (raises
// rather than exporting one that's too close to overflow) — see that
// function's docstring. shift_l1/shift_l2 are PER-LAYER, chosen the same
// way (not hardcoded — a fixed Q15 rescale collapses to near-zero
// precision for layer 1 at this bit-width, see the module docstring in
// export_weights_quantised.py). `(int32_t)w * x[j]` (not
// `(int32_t)w * (int32_t)x[j]`) is deliberate: both operands are 16-bit, so
// avr-gcc pattern-matches this as a widening 16x16->32 multiply instead of
// a full 32x32 one — same trick the docs' "~15 cycles" estimate assumes.

static void policy_forward(const float obs[POLICY_OBS_DIM], float* action)
{
    int16_t x[POLICY_OBS_DIM];
    int16_t h1[POLICY_HIDDEN_DIM];
    int16_t h2[POLICY_HIDDEN_DIM];

    const int32_t max_int = (1L << (POLICY_WEIGHT_BITS - 1)) - 1;
    const int32_t round_l1 = (POLICY_RESCALE_SHIFT_L1 > 0)
        ? (1L << (POLICY_RESCALE_SHIFT_L1 - 1)) : 0L;
    const int32_t round_l2 = (POLICY_RESCALE_SHIFT_L2 > 0)
        ? (1L << (POLICY_RESCALE_SHIFT_L2 - 1)) : 0L;

    for (int j = 0; j < POLICY_OBS_DIM; j++)
    {
        float inv_s = pgm_read_float(&POLICY_INV_SCALE_OBS_IN[j]);
        float q = obs[j] * inv_s;
        long qi = (long)(q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi >  max_int) qi =  max_int;
        if (qi < -max_int) qi = -max_int;
        x[j] = (int16_t)qi;
    }

    for (int i = 0; i < POLICY_HIDDEN_DIM; i++)
    {
        int32_t accum = (int32_t)pgm_read_dword(&POLICY_B1[i]);
        for (int j = 0; j < POLICY_OBS_DIM; j++)
        {
            int16_t w = (int16_t)pgm_read_word(&POLICY_W1[i][j]);
            accum += (int32_t)w * x[j];
        }
        int16_t m = (int16_t)pgm_read_word(&POLICY_RESCALE_M_L1[i]);
        int32_t scaled = (accum * (int32_t)m + round_l1) >> POLICY_RESCALE_SHIFT_L1;
        if (scaled > max_int) scaled = max_int;
        if (scaled < 0)       scaled = 0;   // ReLU
        h1[i] = (int16_t)scaled;
    }

    for (int i = 0; i < POLICY_HIDDEN_DIM; i++)
    {
        int32_t accum = (int32_t)pgm_read_dword(&POLICY_B2[i]);
        for (int j = 0; j < POLICY_HIDDEN_DIM; j++)
        {
            int16_t w = (int16_t)pgm_read_word(&POLICY_W2[i][j]);
            accum += (int32_t)w * h1[j];
        }
        int16_t m = (int16_t)pgm_read_word(&POLICY_RESCALE_M_L2[i]);
        int32_t scaled = (accum * (int32_t)m + round_l2) >> POLICY_RESCALE_SHIFT_L2;
        if (scaled > max_int) scaled = max_int;
        if (scaled < 0)       scaled = 0;
        h2[i] = (int16_t)scaled;
    }

    int32_t accum = (int32_t)pgm_read_dword(&POLICY_B3[0]);
    for (int j = 0; j < POLICY_HIDDEN_DIM; j++)
    {
        int16_t w = (int16_t)pgm_read_word(&POLICY_W3[0][j]);
        accum += (int32_t)w * h2[j];
    }
    float dequant = pgm_read_float(&POLICY_DEQUANT_L3[0]);
    float y = (float)accum * dequant;
    *action = tanhf(y);
}

#else  // float path below

// -----------------------------------------------------------------------------
// Float forward pass — production default.

static void policy_forward(const float obs[POLICY_OBS_DIM], float* action)
{
    float h1[POLICY_HIDDEN_DIM];
    float h2[POLICY_HIDDEN_DIM];

    // Layer 1: obs (POLICY_OBS_DIM) -> h1 (H), ReLU.
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

#endif  // POLICY_QUANTISED_INT8 / POLICY_QUANTISED_INT16

// =============================================================================
// STATE MACHINE
// =============================================================================

static void prime_initial_state()
{
    // Mirror a fresh env reset(): prev_action=0, and the frame stack filled
    // with FRAME_STACK copies of the current (just-zeroed) raw observation
    // — matches FrameStacker.reset()/pendulum_env.py's reset(), no
    // fabricated history.
    prev_action = 0.0f;
    float theta = wrap_pi(0.0f - (float)PI);  // phi=0 (hanging) -> theta=-pi
    float raw_obs[RAW_OBS_DIM] = {
        0.0f,          // motor_pos
        sinf(theta),
        cosf(theta),
        0.0f,          // motor_vel
        0.0f,          // pen_vel
        0.0f,          // prev_action
    };
    frame_stacker_reset(raw_obs);
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
    // Start in zero-accel "hold current speed" state — matches
    // LowLevelServer's CMD_ENGAGE_MOTOR. The stepper is at rest right
    // after enableOutputs(), so this leaves it at rest until the first
    // control tick commands a real accel.
    stepper->moveByAcceleration(0, true);
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

static void control_tick()
{
    // 1. Read the latest ring-buffer sample (self-consistent (pos, vel)
    // pair — same pattern as LowLevelServer's sendState()).
    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1) % BUFFER_SIZE);
    float motor_pos = (float)motor_step_buf[newest] * RAD_PER_STEP;
    float phi = pen_rad_buf[newest];

    // 2. Hard-limit safety: trip back to WAITING if motor strayed past the
    // mechanical envelope. The brake in step 6 is supposed to keep us
    // inside SAFE_LIMIT, but trust nothing — this is the second line of
    // defense.
    if (fabs(motor_pos) > MOTOR_HARD_LIMIT_RAD)
    {
        transition_to_waiting();
        return;
    }

    float motor_vel, pen_vel;
    computeVelocities(&motor_vel, &pen_vel);

    // 3. Build raw observation, push into the frame stack, get the
    // stacked (POLICY_OBS_DIM) input.
    float theta = wrap_pi(phi - (float)PI);
    float raw_obs[RAW_OBS_DIM] = {
        motor_pos, sinf(theta), cosf(theta), motor_vel, pen_vel, prev_action,
    };
    frame_stacker_push(raw_obs);
    float stacked_obs[POLICY_OBS_DIM];
    frame_stacker_get(stacked_obs);

    // 4. Forward pass.
    float action;
    policy_forward(stacked_obs, &action);
    if (action > 1.0f) action = 1.0f;
    else if (action < -1.0f) action = -1.0f;
    last_action = action;

    // 5. Accel-mode action -> commanded angular accel.
    float accel_cmd = action * MAX_ACCEL_RAD_S2;

    // 6. Position-limit safety: past the rail, ignore the policy and brake
    // with full authority back toward center, unconditionally (matches
    // LowLevelServer.ino's CMD_SET_ACCEL — see the 2026-07-05 incident
    // note at the top of this file). Zeroing accel instead would be
    // dangerous: with allow_reverse=true, accel=0 means "hold current
    // speed", so a motor already coasting outward would sail on.
    if (motor_pos >= MOTOR_SAFE_LIMIT_RAD)
    {
        accel_cmd = -MAX_ACCEL_RAD_S2;
    }
    else if (motor_pos <= -MOTOR_SAFE_LIMIT_RAD)
    {
        accel_cmd = MAX_ACCEL_RAD_S2;
    }

    // 7. Command it. allow_reverse=true lets the library decelerate
    // smoothly through zero when the sign of accel opposes the current
    // velocity — no state machine needed on our side.
    int32_t accel_steps_s2 = (int32_t)(accel_cmd * STEPS_PER_RAD);
    stepper->moveByAcceleration(accel_steps_s2, true);

    prev_action = action;
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
    ltoa((long)(pen_position_rad * 1000.0f), p, 10); p += strlen(p); *p++ = ',';
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
    int8_t rc_speed = stepper->setSpeedInUs(MOTOR_MIN_STEP_US);
    if (rc_speed != 0)
    {
        // A silent rejection would leave the stepper unable to issue any
        // pulses. Print a diagnostic and halt rather than booting into a
        // dead-motor mode.
        Serial.print(F("[FATAL] FastAccelStepper config rejected: speed_rc="));
        Serial.println(rc_speed);
        digitalWrite(LED_BUILTIN, HIGH);
        while (true) {}
    }
    stepper->disableOutputs();

    while (!as5600.detectMagnet())
    {
        delay(500);
    }

    last_sample_us = micros();

    // Encoder zero is captured at engage time, not here — see
    // reset_pendulum_tracking() / transition_to_running().

    // Forward-pass self-test: compute the action for a fixed reference obs
    // (a freshly-reset frame stack — 3 identical copies of a single raw
    // frame, matching FrameStacker.reset()) and print it. Compare against
    // the PyTorch student's prediction for the same pose to confirm
    // PROGMEM access + indexing are correct. Re-derive expected values
    // from the .pt file with the helper in docs/end_to_end_runbook.md
    // (step 6) — values are policy-specific and change every distill.
    {
        float raw_hang[RAW_OBS_DIM] = {0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f};  // hanging-down, still
        float raw_up[RAW_OBS_DIM]   = {0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f};  // upright, still
        float test_obs[POLICY_OBS_DIM];
        float test_act;

        for (int f = 0; f < FRAME_STACK; f++)
            for (int j = 0; j < RAW_OBS_DIM; j++)
                test_obs[f * RAW_OBS_DIM + j] = raw_hang[j];
        policy_forward(test_obs, &test_act);
        Serial.print(F("[boot] policy(hanging) = "));
        Serial.println(test_act, 6);

        for (int f = 0; f < FRAME_STACK; f++)
            for (int j = 0; j < RAW_OBS_DIM; j++)
                test_obs[f * RAW_OBS_DIM + j] = raw_up[j];
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
    unsigned long now_us = micros();

    // High-rate encoder/position sampling, independent of control-tick
    // pacing — keeps the ring buffer warm whether or not we're RUNNING.
    if ((unsigned long)(now_us - last_sample_us) >= SAMPLE_PERIOD_US)
    {
        last_sample_us = now_us;
        sampleState();
    }

    // FastAccelStepper drives stepping from a Timer1 ISR — the main loop
    // just paces control ticks at the configured rate.
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
        control_tick();
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
