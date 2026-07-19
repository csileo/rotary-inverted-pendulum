/**
 * PIDControl.ino - Self-contained PID balancing controller for rotary inverted pendulum
 *
 * ARCHITECTURE OVERVIEW
 * ---------------------
 * This sketch runs a complete pendulum balancing system on an Arduino Nano without
 * requiring a computer connection. The control loop runs at a fixed 1 kHz rate.
 *
 * State Machine:
 *   WAITING   - Motor disabled, waiting for pendulum to be lifted near vertical
 *   BALANCING - Motor engaged, PID actively controlling to keep pendulum upright
 *
 * Control Flow (each 1ms iteration):
 *   1. Read encoder via I2C (AS5600, 400 kHz)
 *   2. Apply low-pass filter to angle
 *   3. Compute PID output (P + I + D terms)
 *   4. Update motor target position
 *   5. Run AccelStepper to move toward target
 *
 * TIMING DESIGN
 * -------------
 * - Control loop: Fixed 1 kHz (1000 μs period). If loop() is called before 1ms has
 *   elapsed, it returns immediately without doing work. This ensures consistent dt.
 * - I2C clock: 400 kHz (vs default 100 kHz) - reduces encoder read from 650μs to 290μs
 * - Serial output: 100 Hz (every 10ms), uses buffered integer output to avoid blocking
 * - Overrun detection: Flags iterations that exceed 1.5x expected period
 *
 * SERIAL OUTPUT FORMAT
 * --------------------
 * When enabled (press 'P'), outputs CSV at 100 Hz:
 *   motor_pos, motor_target, pendulum×1000, state, P×1000, I×1000, D×1000, freq_hz, overruns
 *
 * Values are transmitted as integers (×1000) to avoid slow float-to-string conversion.
 * The companion Julia script (collect_and_plot.jl) divides by 1000 when parsing.
 *
 * Why integers? Serial.print(float) takes ~500μs on Arduino Nano (ATmega328P) because
 * there's no hardware floating-point unit, so float-to-string conversion is done in
 * slow software. Using ltoa() with integers takes ~50μs instead, preventing overruns.
 *
 * MOTOR CENTERING
 * ---------------
 * To prevent the motor from drifting to its ±90° limits, we bias the pendulum target
 * angle based on motor position: target = upright + K_MOTOR_CENTERING * motor_pos
 *
 * This creates a slight tilt that naturally pushes the motor back toward center.
 * The bias is small enough (~0.6° per 30° motor offset) to not destabilize balancing.
 */

#include <AccelStepper.h>
#include <AS5600.h>
#include <Wire.h>

// =============================================================================
// PIN DEFINITIONS
// =============================================================================
const int DIR_PIN = 2;
const int STEP_PIN = 9;     // Timer1 OC1A (kept on this pin so FastAccelStepper-based sketches work too)
const int ENABLE_PIN = 5;

// =============================================================================
// HARDWARE CONSTANTS
// =============================================================================
const long STEPS_PER_REVOLUTION = 200 * 8;  // 200 steps * 8 microsteps = 1600
const float DEGREES_PER_STEP = 360.0f / STEPS_PER_REVOLUTION;
const float STEPS_PER_DEGREE = STEPS_PER_REVOLUTION / 360.0f;

// =============================================================================
// COMMUNICATION SETTINGS
// =============================================================================
const long SERIAL_BAUD_RATE = 500000;      // High speed for data collection
const long I2C_CLOCK_HZ = 400000;          // Fast mode I2C (vs 100kHz default)

// =============================================================================
// MOTOR SETTINGS
// =============================================================================
// AccelStepper on Arduino Nano is limited to ~4000 steps/sec due to step pulse
// timing overhead. High acceleration ensures motor reaches max speed in ~40ms
// (4000 / 100000 = 0.04s), which is fast compared to typical balance recovery.
const long MOTOR_MAX_SPEED = 4000;         // steps/sec (practical limit on Nano)
const long MOTOR_ACCELERATION = 100000;    // steps/sec² (high = instant response)

// =============================================================================
// CONTROL PARAMETERS (tune these)
// =============================================================================
//
// TUNING GUIDE - Recommended order:
//   1. K_MOTOR_CENTERING - Prevent motor drift to limits
//   2. Kp - Get basic responsiveness (start low, increase until oscillation)
//   3. Kd - Add damping to reduce oscillation
//   4. Ki - Eliminate steady-state error (use sparingly)
//   5. ENGAGEMENT_MARGIN_DEG - Adjust catch window if needed
//   6. Fine-tune limits (INTEGRAL_LIMIT, PID_OUTPUT_LIMIT_DEG)
//
// =============================================================================

// PID gains
// -----------------------------------------------------------------------------
// Kp (Proportional): Reacts to current error
//   - Increase if: pendulum responds too slowly, falls before correcting
//   - Decrease if: pendulum oscillates rapidly, motor is jerky
//   - Typical range: 0.5 - 2.0
const float Kp = 0.30f;

// Ki (Integral): Accumulates error over time, eliminates steady-state offset
//   - Increase if: pendulum settles with persistent tilt
//   - Decrease if: motor drifts, overshoots, or oscillates slowly
//   - Typical range: 1.0 - 6.0 (use sparingly, often the cause of drift)
const float Ki = 0.1f;

// Kd (Derivative): Reacts to rate of change, provides damping
//   - Increase if: pendulum oscillates, needs more damping
//   - Decrease if: motor is jerky or reacts to noise
//   - Typical range: 0.005 - 0.05
const float Kd = 0.005f;

// Filter cutoff for derivative calculation
// -----------------------------------------------------------------------------
//   - Increase if: response feels sluggish
//   - Decrease if: derivative term is noisy/jerky
//   - Must be less than half of CONTROL_FREQUENCY_HZ (Nyquist)
const float FILTER_CUTOFF_HZ = 150.0f;

// Motor centering - prevents motor from drifting to position limits
// -----------------------------------------------------------------------------
// Biases pendulum target angle based on motor position to push motor back
// Effect: motor_pos * K = degrees of pendulum tilt toward center
//   - Increase if: motor drifts toward ±90° limits during balancing
//   - Decrease if: centering interferes with balancing (causes oscillation)
//   - At 0.05: 30° motor offset → 1.5° pendulum tilt toward center
//   - At 0.10: 30° motor offset → 3.0° pendulum tilt toward center
//   - Typical range: 0.03 - 0.15
const float K_MOTOR_CENTERING = 0.05f;

// Engagement window - how close to vertical before engaging motor
// -----------------------------------------------------------------------------
//   - Increase if: want to catch pendulum earlier, more aggressive recovery
//   - Decrease if: controller struggles when pendulum is far from vertical
//   - Typical range: 15 - 40 degrees
const float ENGAGEMENT_MARGIN_DEG = 25.0f;

// Control loop timing
// -----------------------------------------------------------------------------
const int CONTROL_FREQUENCY_HZ = 500;
const unsigned long CONTROL_PERIOD_US = 1000000UL / CONTROL_FREQUENCY_HZ;

// Limits
// -----------------------------------------------------------------------------
// Motor position limits (degrees from starting position)
//   - Set based on physical constraints (wire wrap, frame collision)
const float MOTOR_LIMIT_DEG = 90.0f;

// PID output limit per cycle (degrees of motor movement)
//   - Increase if: motor can't keep up with fast disturbances
//   - Decrease if: motor movements are too aggressive
const float PID_OUTPUT_LIMIT_DEG = 50.0f;

// Integral windup limit (degrees * seconds accumulated)
//   - Increase if: integral never reaches full effect
//   - Decrease if: integral causes large overshoots after disturbances
const float INTEGRAL_LIMIT = 50.0f;

// =============================================================================
// STATE VARIABLES
// =============================================================================
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
AS5600 as5600;

enum State { WAITING, BALANCING };
State state = WAITING;

// Pendulum state
float pendulum_actual_deg = 0.0f;
float pendulum_target_deg = 180.0f;

// Motor state
float motor_target_pos = 0.0f;  // in steps

// PID state
float pid_integral = 0.0f;
float pid_prev_error = 0.0f;

// PID debug (last computed values for diagnostics)
float debug_p_term = 0.0f;
float debug_i_term = 0.0f;
float debug_d_term = 0.0f;

// Loop timing monitoring
unsigned int loop_overruns = 0;

// Timing
unsigned long prev_time_us = 0;

// Serial output
bool print_enabled = false;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

long degreesToSteps(float degrees)
{
    return (long)(degrees * STEPS_PER_DEGREE);
}

float stepsToDegrees(long steps)
{
    return steps * DEGREES_PER_STEP;
}

/**
 * Compute exponential low-pass filter coefficient from cutoff frequency.
 * alpha = 1.0 means infinite smoothing (keep old value, ignore new)
 * alpha = 0.0 means no filtering (use new value directly)
 */
float computeFilterAlpha(float cutoff_hz, float dt_s)
{
    float omega = 2.0f * PI * cutoff_hz;
    float alpha = (1.0f - omega * dt_s / 2.0f) / (1.0f + omega * dt_s / 2.0f);
    return constrain(alpha, 0.0f, 1.0f);
}

/**
 * Read pendulum angle with multi-revolution tracking.
 * Returns cumulative angle in degrees.
 */
float readPendulumAngle()
{
    const long AS5600_RESOLUTION = 4096;
    const long WRAPAROUND_THRESHOLD = AS5600_RESOLUTION / 2;
    const float DEG_PER_SEGMENT = 360.0f / AS5600_RESOLUTION;

    static long raw_prev = 0;
    static bool first_reading = true;
    static float position = 0.0f;

    long raw = as5600.rawAngle();

    if (first_reading)
    {
        raw_prev = raw;
        first_reading = false;
    }

    long delta = raw - raw_prev;

    // Handle wraparound (movement > 180 deg between samples)
    if (delta > WRAPAROUND_THRESHOLD)  delta -= AS5600_RESOLUTION;
    if (delta < -WRAPAROUND_THRESHOLD) delta += AS5600_RESOLUTION;

    position += delta * DEG_PER_SEGMENT;
    raw_prev = raw;

    return position;
}

/**
 * Find the closest upright target angle (180, 540, -180, etc.)
 */
float findClosestUprightTarget(float current_angle)
{
    int revs = (int)(current_angle / 360.0f);
    float sign = (current_angle >= 0) ? 1.0f : -1.0f;
    return 180.0f * sign + 360.0f * revs;
}

// =============================================================================
// SERIAL COMMANDS
// =============================================================================

void handleSerialCommands()
{
    if (!Serial.available()) return;

    char cmd = Serial.read();
    // Flush any remaining bytes
    while (Serial.available()) Serial.read();

    switch (cmd)
    {
        case 'P':
        case 'p':
            print_enabled = !print_enabled;
            break;
        case 'M':
        case 'm':
            printMagnetInfo();
            break;
        case 'R':
        case 'r':
            resetPIDState();
            Serial.println("PID state reset");
            break;
    }
}

void printMagnetInfo()
{
    Serial.print("[AS5600] Magnet: ");
    if (as5600.magnetTooWeak()) Serial.println("TOO WEAK");
    else if (as5600.magnetTooStrong()) Serial.println("TOO STRONG");
    else Serial.println("OK");

    Serial.print("[AS5600] Magnitude: ");
    Serial.println(as5600.readMagnitude());
}

void printPlotData()
{
    static unsigned long last_print_ms = 0;
    static unsigned long last_print_us = 0;
    static unsigned int loop_count = 0;
    static bool was_enabled = false;
    const unsigned long PRINT_INTERVAL_MS = 10;

    loop_count++;

    if (!print_enabled)
    {
        was_enabled = false;
        return;
    }

    // Reset timing state when print is first enabled to avoid outliers
    if (!was_enabled)
    {
        was_enabled = true;
        last_print_ms = millis();
        last_print_us = micros();
        loop_count = 0;
        return;
    }

    unsigned long now = millis();
    if (now - last_print_ms >= PRINT_INTERVAL_MS)
    {
        last_print_ms = now;

        // Compute actual frequency from elapsed microseconds
        unsigned long now_us = micros();
        unsigned long elapsed_us = now_us - last_print_us;
        last_print_us = now_us;
        unsigned int freq_hz = (unsigned long)loop_count * 1000000UL / elapsed_us;

        // Format: motor_pos,motor_target,pendulum_x1000,state,p_x1000,i_x1000,d_x1000,freq_hz,overruns
        // Build in buffer using ltoa (faster and smaller than snprintf)
        char buf[80];
        char* p = buf;

        ltoa(stepper.currentPosition(), p, 10); p += strlen(p); *p++ = ',';
        ltoa((long)motor_target_pos, p, 10); p += strlen(p); *p++ = ',';
        ltoa((long)(pendulum_actual_deg * 1000), p, 10); p += strlen(p); *p++ = ',';
        *p++ = state == BALANCING ? '1' : '0'; *p++ = ',';
        ltoa((long)(debug_p_term * 1000), p, 10); p += strlen(p); *p++ = ',';
        ltoa((long)(debug_i_term * 1000), p, 10); p += strlen(p); *p++ = ',';
        ltoa((long)(debug_d_term * 1000), p, 10); p += strlen(p); *p++ = ',';
        utoa(freq_hz, p, 10); p += strlen(p); *p++ = ',';
        utoa(loop_overruns, p, 10); p += strlen(p);
        *p = '\0';

        Serial.println(buf);
        loop_count = 0;
    }
}

// =============================================================================
// PID CONTROLLER
// =============================================================================

void resetPIDState()
{
    pid_integral = 0.0f;
    pid_prev_error = 0.0f;
}

/**
 * Compute PID output and update motor target.
 * Returns the computed output (in steps).
 */
void updatePIDControl(float dt_s)
{
    // Calculate error
    float error = pendulum_target_deg - pendulum_actual_deg;

    // Proportional term
    float p_term = Kp * error;

    // Integral term with anti-windup
    pid_integral += error * dt_s;
    pid_integral = constrain(pid_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);
    float i_term = Ki * pid_integral;

    // Derivative term
    float derivative = (error - pid_prev_error) / dt_s;
    pid_prev_error = error;
    float d_term = Kd * derivative;

    // Save for diagnostics
    debug_p_term = p_term;
    debug_i_term = i_term;
    debug_d_term = d_term;

    // Compute total output (in degrees, then convert to steps)
    float output_deg = p_term + i_term + d_term;

    // Limit output per cycle
    output_deg = constrain(output_deg, -PID_OUTPUT_LIMIT_DEG, PID_OUTPUT_LIMIT_DEG);

    // Convert to steps and add to current position
    float output_steps = output_deg * STEPS_PER_DEGREE;
    motor_target_pos = stepper.currentPosition() + output_steps;

    // Clamp motor position to limits
    long max_steps = degreesToSteps(MOTOR_LIMIT_DEG);
    if (motor_target_pos > max_steps)
    {
        motor_target_pos = max_steps;
        // Anti-windup: prevent integral from pushing further into limit
        if (pid_integral > 0) pid_integral *= 0.9f;
    }
    else if (motor_target_pos < -max_steps)
    {
        motor_target_pos = -max_steps;
        if (pid_integral < 0) pid_integral *= 0.9f;
    }
}

// =============================================================================
// STATE MACHINE
// =============================================================================

void transitionToBalancing()
{
    state = BALANCING;
    stepper.enableOutputs();
    motor_target_pos = stepper.currentPosition();

    // Initialize PID state properly to avoid derivative spike
    pid_integral = 0.0f;
    // Set prev_error to current error so first derivative is 0, not a huge spike
    pid_prev_error = pendulum_target_deg - pendulum_actual_deg;

    // Clear debug values
    debug_p_term = 0.0f;
    debug_i_term = 0.0f;
    debug_d_term = 0.0f;
    // Serial.println("BALANCING");
}

void transitionToWaiting()
{
    state = WAITING;
    stepper.stop();
    stepper.disableOutputs();
    resetPIDState();
    // Serial.println("WAITING");
}

// =============================================================================
// LED FEEDBACK
// =============================================================================

void updateLED()
{
    static unsigned long last_toggle_ms = 0;
    static bool led_on = false;

    unsigned long period_ms = print_enabled ? 500 : 100;
    unsigned long now = millis();

    if (now - last_toggle_ms >= period_ms)
    {
        last_toggle_ms = now;
        led_on = !led_on;
        digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
    }
}

// =============================================================================
// SETUP
// =============================================================================

void setup()
{
    Serial.begin(SERIAL_BAUD_RATE);
    Wire.begin();
    Wire.setClock(I2C_CLOCK_HZ);
    as5600.begin();

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Configure stepper
    stepper.setMaxSpeed(MOTOR_MAX_SPEED);
    stepper.setAcceleration(MOTOR_ACCELERATION);
    stepper.setEnablePin(ENABLE_PIN);
    stepper.setPinsInverted(false, false, true);  // Enable pin is active-low

    // Wait for encoder magnet
    while (!as5600.detectMagnet())
    {
        delay(1000);
    }

    // Initialize pendulum angle reading
    pendulum_actual_deg = readPendulumAngle();

    // Initialize timing to avoid huge dt on first loop iteration
    prev_time_us = micros();

    digitalWrite(LED_BUILTIN, LOW);
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop()
{
    // Run stepper on EVERY iteration to achieve full speed
    // (run() only executes one step per call, so it needs to be called frequently)
    stepper.run();

    // Fixed-rate timing: wait until control period has elapsed
    unsigned long now_us = micros();
    unsigned long elapsed_us = now_us - prev_time_us;
    if (elapsed_us < CONTROL_PERIOD_US)
    {
        return;  // Not time yet, exit early (but stepper.run() was still called above)
    }

    // Track overruns (loop took >1.5x longer than expected)
    if (elapsed_us > CONTROL_PERIOD_US * 3 / 2)
    {
        loop_overruns++;
    }

    // Calculate actual dt (should be ~CONTROL_PERIOD_US)
    float dt_s = elapsed_us * 1e-6f;
    prev_time_us = now_us;

    // Handle serial commands
    handleSerialCommands();

    // Update LED
    updateLED();

    // Read and filter pendulum angle
    float alpha = computeFilterAlpha(FILTER_CUTOFF_HZ, dt_s);
    float raw_angle = readPendulumAngle();
    pendulum_actual_deg = alpha * pendulum_actual_deg + (1.0f - alpha) * raw_angle;

    // Find closest upright target, biased by motor position to prevent drift
    float nearest_upright_deg = findClosestUprightTarget(pendulum_actual_deg);
    float motor_pos_deg = stepsToDegrees(stepper.currentPosition());
    float centering_bias_deg = K_MOTOR_CENTERING * motor_pos_deg;
    pendulum_target_deg = nearest_upright_deg + centering_bias_deg;

    // Check if pendulum is close to vertical (use unbiased target for engagement decision)
    float error_from_vertical = fabs(nearest_upright_deg - pendulum_actual_deg);
    bool close_to_vertical = (error_from_vertical <= ENGAGEMENT_MARGIN_DEG);

    // State machine transitions
    if (state == WAITING && close_to_vertical)
    {
        transitionToBalancing();
    }
    else if (state == BALANCING && !close_to_vertical)
    {
        transitionToWaiting();
    }

    // Control loop
    if (state == BALANCING)
    {
        updatePIDControl(dt_s);
        stepper.moveTo((long)motor_target_pos);
    }

    // Output data for plotting
    printPlotData();
}
