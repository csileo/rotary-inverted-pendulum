# Transport delay: how it shrank, and how we measure it

This rig used to have ~50 ms of laptop-to-motor transport delay. It's
now ~14 ms. The path there was three fixes, each attacking a different
layer of the pipeline. None of the three was about "delay" *per se*;
they targeted other problems and the delay drop was the by-product.

## The pipeline

```
policy(obs) -> action  ┐
                       │  (1) USB serial          ~1–5 ms
                       ▼
                   Arduino loop  -- (2) Cmd dispatch
                                 -- (3) Stepper driver / ISR
                                 -- (4) Motor mechanical response
                       │
                       ▼
                  AS5600 encoder -- (5) I²C read       ~5 ms
                       │
                       ▼
                 policy obs(t+1)
```

Total round-trip transport delay = (1) + (2) + (3) + (4) + (5).

## The three fixes (in chronological order)

| Date       | Fix                                                                      | Layer it targets | Why                                                                                  | Side effect on delay                                                                            |
|------------|--------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| 2026-05-03 | `AccelStepper` → `FastAccelStepper` (commit `c46480d`)                   | (3) stepper      | Polled step-pulse timing in AccelStepper jittered above 50 k steps/s², causing step-skipping; firmware position diverged from reality. FastAccelStepper drives STEP from Timer1 OC1A ISR — honest pulses. Required moving STEP_PIN to pin 9. | Slight: ISR timing is more deterministic than polled, so jitter on layer (3) shrinks. Median delay roughly unchanged. |
| 2026-05-16 | Action semantics: position-target → angular-acceleration (commit `396c5d4`) | (3) stepper      | Position commands forced a fresh trapezoidal land-on-target ramp each control tick (10–30 ms of mechanical lag per tick). Closed-loop sim-vs-real diverged at resonance; sim_upright=0.14 vs real_upright=0.73 on identical action sequences. Switching to `moveByAcceleration(int32 steps_s2, allow_reverse=true)` lets the motor integrate accel continuously across ticks, with smooth zero-crossing direction reversal. | **Big drop.** Removes the per-tick targeting ramp (10–30 ms). This is the dominant contributor to the 50 ms → ~14 ms reduction. |
| 2026-05-16 | Observation extended with `prev_action` (commit `cae2a1b`)               | (1) policy       | Even after fixes 1 + 2 the remaining ~14 ms makes the system a POMDP from the policy's point of view — it can't tell whether obs(t) reflects action(t) or action(t-1). Adding `prev_action` to the obs restores the Markov property for the action pipeline. | None directly. Doesn't change the physical delay; lets the policy reason about it. |

## Current measurement (2026-05-16)

Two methods, same conclusion.

### Method 1 — sysid_accel step test (pendulum held)

Recorded by `sysid_accel.py step` at 200 Hz logging while driving the
firmware directly via `set_acceleration` calls. Motor responds within
**one 200 Hz sample (≤ 5 ms)** of an accel-command step change. This
measures layers (1) + (2) + (3) + (4) without the I²C/Python read leg.

### Method 2 — real-rig deploy log half-step model fit

From `run_policy.py --log /tmp/pdfix.npz` running at the policy's 35 Hz
control rate. Pick a step where the commanded accel changes sharply and
look at the velocity delta two steps later:

```
idx 91: prev cmd = -37.5,  cmd = -149,  observed Δv = -2.69 rad/s
        expected if 0-step delay         -4.25
        expected if 1-step delay         -1.07
        expected if ½-step delay  ✓      -2.66
```

Fits a **½-control-step delay model** to within filter noise. At 35 Hz
control, that's **≈ 14 ms** of effective transport delay end-to-end —
including the encoder read and Python decision time that method 1
skips.

## Implications

- **DR ranges in `pendulum_env.py` are still position-mode calibrated.**
  `DR_ACTION_DELAY_STEPS_RANGE = (1, 3)` was set when the real delay was
  ~50 ms (1–3 steps at 35 Hz). Post-accel-mode reality is ~½ step.
- Integer-step delay DR (sample 0 or 1) is a coarse fit to a fractional
  delay. A continuous **action-lag** DR (first-order filter with random
  tau ∈ ~[5, 20] ms) matches the real layer (3)+(4) dynamics more
  directly and gives the optimiser a smoother gradient than 0-or-1
  discrete sampling.
- Curriculum stage 2/3 delay ranges should be tightened to bracket the
  actual ~14 ms, not the historical 30–50 ms.
