# PID Tuning History

This document tracks the iterative tuning process for the PID balancing controller.

## Hardware Setup
- **Motor:** NEMA17 stepper with DRV8825 driver, 8x microstepping (1600 steps/rev)
- **Encoder:** AS5600 magnetic encoder (12-bit, 4096 counts/rev)
- **Controller:** Arduino Nano (ATmega328P, 16MHz)
- **Motor limits:** ±90° from starting position (wire constraint, no slip ring)

## Control Architecture
- State machine: WAITING → BALANCING (engages within ±25° of vertical)
- PID output controls motor position (not velocity)
- Low-pass filter on pendulum angle before PID calculation
- Anti-windup on integral term with limit and decay at motor limits

---

## Iteration 1: Original Ziegler-Nichols Values

**Date:** 2025-01-10

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Kp | 1.2 (0.6 × Ku, where Ku=2.0) |
| Ki | 24.0 (2 × Kp / Tu, where Tu=0.1) |
| Kd | 0.015 (Kp × Tu / 8) |
| Filter | 500 Hz |
| Motor MaxSpeed | 200000 |
| Motor Accel | 100000 |

**Bugs found in code review:**
- Integer division bug: `controlPeriod = 1/1000 = 0` (always true condition)
- No integral anti-windup
- Motor limit just froze instead of clamping
- Tare function did nothing

**Observations:**
- Motor drifted heavily toward ±90° limits
- Hit saturation and lost balance
- Balanced for ~6 seconds before falling
- Integral wind-up caused persistent drift

---

## Iteration 2: Reduced Ki, Increased Kd

**Date:** 2025-01-10

**Parameters:**
| Parameter | Value | Change |
|-----------|-------|--------|
| Kp | 1.2 | - |
| Ki | 6.0 | ↓ from 24.0 |
| Kd | 0.03 | ↑ from 0.015 |
| Filter | 500 Hz | - |

**Observations:**
- Motor no longer saturating at ±90° (stayed within ±60°)
- Motor returns toward center after falls
- Still very oscillatory during balance attempts
- Rapid oscillations ~±20-30° not decaying
- System on edge of stability

**Diagnosis:** Insufficient damping, derivative amplifying high-frequency noise

---

## Iteration 3: More Damping, Lower Filter

**Date:** 2025-01-10

**Parameters:**
| Parameter | Value | Change |
|-----------|-------|--------|
| Kp | 1.0 | ↓ from 1.2 |
| Ki | 6.0 | - |
| Kd | 0.08 | ↑ from 0.03 |
| Filter | 100 Hz | ↓ from 500 Hz |

**Observations:**
- High-frequency oscillations eliminated (filter working)
- Motor stays within ±60° (no saturation)
- Best stable period at end of run with ~±10-15° error
- Still falling and needing recovery multiple times
- Slower oscillations remain during balance
- Clear improvement over previous iterations

**Diagnosis:** Better, but still needs more damping or less aggression

---

## Iteration 4: Bug Fix + Enhanced Diagnostics

**Date:** 2025-01-10

**Bug fixes applied:**
- Fixed first-loop timing bug: `prev_time_us` now initialized in `setup()` to avoid huge dt on first iteration
- This was causing derivative spikes and filter miscalculation on startup

**Enhanced data collection:**
- Now collecting individual PID terms (P, I, D) for diagnosis
- Added state (WAITING/BALANCING) to output
- New plot: `plot_pid_terms.png` shows each term over time

**Parameters:** Same as Iteration 3 (Kp=1.0, Ki=6.0, Kd=0.08, Filter=100Hz)

**Next:** Run with bug fixes to establish new baseline, then continue tuning.

---

## Iteration 5: Timing Optimization & Fixed 1 kHz Loop

**Date:** 2025-01-10

**Problem:** Loop timing was inconsistent, varying widely based on what code executed each iteration. Serial output was causing significant delays and false overrun detections.

**Changes:**

1. **Fixed-rate control loop (1 kHz)**
   - Implemented early-return pattern: loop exits immediately if <1000μs elapsed
   - Ensures consistent dt for PID calculations
   - Added overrun detection (flags iterations >1.5x expected period)

2. **I2C speed increase (100 kHz → 400 kHz)**
   - AS5600 encoder read reduced from ~650μs to ~290μs
   - This was the main bottleneck consuming 65% of loop time

3. **Serial output optimization**
   - Problem: `Serial.print(float, 2)` takes ~500μs per call due to AVR software float formatting
   - Solution: Transmit values as integers (×1000), decode in Julia
   - Further optimization: Use `ltoa()` + manual buffer concatenation instead of `snprintf`
   - Result: Serial output no longer causes overruns

4. **Baud rate increase (115200 → 500000)**
   - Reduces time spent waiting for TX buffer
   - Arduino Nano supports up to 2 Mbaud, but 500k is reliable

**Serial output comparison:**

| Approach | Overruns | Loop Freq | Flash Size |
|----------|----------|-----------|------------|
| Float Serial.print() ×9 calls | 1065 | 881 Hz | 12,994 B |
| Integer ×1000 Serial.print() ×9 | 0 | 985 Hz | 12,560 B |
| Integer snprintf buffer | 0 | 1001 Hz | 13,904 B |
| Integer ltoa buffer (final) | 0 | 1003 Hz | 12,700 B |

**Final timing results:**
- Loop frequency: 1003 Hz mean (target: 1000 Hz)
- Loop frequency range: 900-1100 Hz
- Overruns: 0

**Parameters:** Kp=0.8, Ki=4.0, Kd=0.015, Filter=100Hz

---

## Next Steps

Timing infrastructure is now solid. Ready to focus on PID tuning with reliable data collection.

---

## Systematic Tuning Approach

1. **Fix bugs first** - ensure system behaves as expected
2. **Collect diagnostic data** - PID terms, state, timing
3. **One parameter at a time** - isolate effects of each change
4. **Quantitative comparison** - use RMS error, balancing time metrics
5. **Document everything** - track what works and what doesn't

---

## Notes

- AccelStepper on Arduino Nano limited to ~4000 steps/sec with `run()`
- Current maxSpeed/acceleration settings (200000/100000) are above this limit but acceleration is high enough to act nearly instant
- Engagement margin of ±25° seems appropriate
- Motor limit of ±90° is a hard constraint due to wiring
