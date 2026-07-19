# Domain Randomization (sim training)

Sim-to-real for this rig is a closed-loop instability problem on a
position-controlled stepper. Anything we miss in the sim model the
policy will silently overfit to. Domain randomization (DR) is the
mechanism we use to bridge that gap: we train against a *distribution*
of plausible rigs rather than a single nominal one, so the policy is
robust to the residual mismatch between our identified parameters and
whatever the real hardware looks like on a given day.

This doc summarises **what is randomized, by how much, and why**, plus
how DR fits into the curriculum schedule and where the knobs live in
code. For the broader RL plan (phases, decisions, status) see
[`../RL_PLAN.md`](../RL_PLAN.md). For the transition-level contract see
[`rl_transitions.md`](rl_transitions.md). For the sysid measurements
that set the bracketed values see [`sysid_runbook.md`](sysid_runbook.md).

## How to enable it

```bash
python train_sac.py --domain-randomization ...
```

Or, equivalently, run the full curriculum:

```bash
./curriculum_train.sh <run-name-prefix>
```

The flag flips a single boolean on the env
([`pendulum_env.py:248`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py#L248)).
When off, the env runs deterministically against the nominal
sysid params — useful for debugging but **never** for the policy that
will be deployed.

The **eval env always stays deterministic** (DR off), so best-model
selection during training tracks performance on the nominal-physics
reference scenario instead of being washed out by sample-to-sample
randomisation noise. See
[`train_sac.py:86`](../RotaryInvertedPendulum-python/src/rl/train_sac.py#L86).

## What is randomized

All ranges are defined as module constants in
[`pendulum_env.py:61-97`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py#L61-L97).
Most are sampled **once per episode** in
[`_sample_dr_params`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py#L384);
the dt-jitter and observation noise are sampled **per step**.

### Physical parameters (per episode)

| Parameter                                   | Range                | Source constant                                  | Why this width                                                                                                                                                                                                                                                      |
| ------------------------------------------- | -------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pendulum mass                               | nominal × (1 ± 0.10) | `DR_PENDULUM_MASS_FRAC = 0.10`                   | Narrowed from ±0.20 after CAD cross-check (Onshape part with PLA-density-corrected materials) agreed with sysid `m·d` to within 1%; mass is a one-shot scale measurement and the remaining uncertainty (unmodelled magnet, infill variation) sits well inside ±10%. |
| Pendulum COM distance                       | nominal × (1 ± 0.10) | `DR_PENDULUM_COM_FRAC = 0.10`                    | Bearing seating and tip mass placement vary; ±10 % covers reasonable rebuilds.                                                                                                                                                                                      |
| Pendulum joint friction (viscous + Coulomb) | nominal × [0.5, 2.0] | `DR_PENDULUM_FRICTION_MULT_RANGE = (0.5, 2.0)`   | Friction depends on grease state, temperature, and bearing seating; same multiplier applied to both terms because they share the bearing as a source.                                                                                                               |
| Motor joint stiction (`frictionloss`)       | [0.0, 0.005] N·m     | `DR_MOTOR_FRICTIONLOSS_RANGE_N_M = (0.0, 0.005)` | Steppers have detent torque that the position actuator doesn't capture; lower bound includes 0 for backward compat with Phase 2 policies trained without stiction.                                                                                                  |

Nominal values come from
[`sysid_params.json`](../RotaryInvertedPendulum-python/src/rl/sysid_params.json),
written by the Phase 0 sysid pipeline. **Pendulum inertia about its own
COM** (`PENDULUM_I_COM_SWING_KG_M2` at
[`pendulum_env.py:71`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py#L71))
is hard-coded from Onshape CAD rather than back-computed from sysid
`I_axis − m·d²` — it's a geometric property of the part, not a quantity
that varies with rebuilds, so it isn't randomised. MuJoCo applies
parallel-axis automatically from `body_ipos`, giving per-episode pivot
inertia `m·d² + I_com_swing`. Previously this was forced to ≈0 (point-
mass approximation); the CAD value (~8.06e-6 kg·m²) adds ~25% to the
effective pivot inertia at nominal, matching the measured `I_axis`.

### Actuator / control-loop realism (per episode)

| Parameter                                  | Range                            | Source constant                    | Why this width                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------ | -------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Motor first-order lag τ                    | [0.0, 0.010] s                   | `DR_MOTOR_TAU_RANGE_S`             | Phase 2.5 fits against real-hardware logs found τ ≈ 0; we still sweep up to 10 ms to keep a robustness margin against load-dependent slowdowns.                                                                                                                                                                                                          |
| Action delay (transport delay on commands) | [4, 7] control steps             | `DR_ACTION_DELAY_STEPS_RANGE`      | At 100 Hz, brackets the measured ~50 ms hardware delay (5 steps) ±1 step margin. Phase 2.6 widened this after lowering `MOTOR_ACCELERATION` (100 k → 50 k steps/s²) on the Arduino. The curriculum scales this to physical ms at lower control rates — see below.                                                                                        |
| Control-step dt jitter                     | n_substeps × (1 ± 0.05) per step | `DR_CONTROL_DT_JITTER_FRAC = 0.05` | Empirically the single most important DR. Without it, SAC at strict timing finds the **active-correction attractor** (motor saws ±0.5 rad even when balanced); with it, SAC finds the **calm minimal-action attractor** that dominates real-world performance. See [`control_rate_selection.md`](control_rate_selection.md) "calm vs active attractors". |

### Observation noise (per step)

| Parameter                   | Range                                 | Source constant              | Why this width                                                                                                                  |
| --------------------------- | ------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Pendulum angle quantisation | snapped to AS5600 LSB (2π / 4096 rad) | `PENDULUM_LSB_RAD`           | Models the encoder's 12-bit resolution. Always applied when DR is on.                                                           |
| Position noise σ            | 0.005 rad                             | `DR_OBS_NOISE_STD_POS_RAD`   | Mimics finite-diff jitter and encoder noise observed on the rig. Added to motor pos and pendulum angle.                         |
| Velocity noise σ            | 0.05 rad/s                            | `DR_OBS_NOISE_STD_VEL_RAD_S` | Velocity is finite-differenced from a noisy position signal; σ chosen to match the observed jitter at this rig's filter cutoff. |

## Curriculum staging

Training in three stages
([`curriculum_train.sh`](../RotaryInvertedPendulum-python/src/rl/curriculum_train.sh))
is more reliable than one shot at full DR width. Each stage `--resume`s
from the last so capabilities accumulate.

The two parameters that are annealed across stages are **action delay**
and **dt jitter**. The other DR ranges are constant for all stages
because the policy benefits from seeing them from the start.

| Stage          | Action delay (physical) | Motor τ    | dt jitter | Steps             |
| -------------- | ----------------------- | ---------- | --------- | ----------------- |
| **1 — easy**   | [0, 20] ms              | [0, 5] ms  | ±20 %     | `STEPS_PER_STAGE` |
| **2 — medium** | [20, 50] ms             | [0, 10] ms | ±10 %     | `STEPS_PER_STAGE` |
| **3 — final**  | [30, 60] ms             | [0, 10] ms | ±5 %      | `STEPS_PER_STAGE` |

Notes:

- The script converts physical-ms ranges into integer step counts at the
  configured `CONTROL_FREQ`. At low rates (e.g. 35 Hz) stages 2 and 3
  can collide after rounding; in that case stage 3 is skipped and the
  final policy is the stage-2 best model. The script logs this clearly.
- The dt-jitter anneal is intentional: high jitter early forces the
  policy out of the active-correction attractor; lower jitter later
  lets it specialise for deployment-realistic timing.
- Stage 3 brackets the hardware's ~50 ms delay with ~one step of margin
  on each side at 35 Hz (the canonical rate for this rig).

## Reset diversity (not strictly DR, but adjacent)

`reset()` itself adds initial-condition diversity that is essential for
the policy to learn recovery from arbitrary starts:

- **Motor start**: uniform in ±0.7 × motor safety limit (≈ ±88°). Keeps
  the reset clear of the ±125° clamp while covering most of the working
  range. Without this, the policy never practises returning from the
  limit and gets stuck there at deploy time
  ([`pendulum_env.py:362-374`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py#L362-L374)).
- **Pendulum start**: hanging-down ± 0.05 rad (small noise around the
  natural rest angle).

This is on regardless of `--domain-randomization` because it's about
training-state coverage, not modelling-error robustness.

## Where the knobs live

| Knob                                | Default                              | Where it's set / read                                                                                                                                                           |
| ----------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Module constants (`DR_*`)           | as above                             | [`pendulum_env.py:61-97`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py#L61-L97)                                                                                      |
| Per-instance overrides (curriculum) | None → fall back to module constants | `RotaryInvertedPendulumEnv.__init__` ([`pendulum_env.py:255-284`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py#L255-L284))                                           |
| CLI overrides                       | unset → module defaults              | `train_sac.py` flags `--dr-tau-min/max`, `--dr-delay-min/max`, `--dr-dt-jitter-frac` ([`train_sac.py:198-226`](../RotaryInvertedPendulum-python/src/rl/train_sac.py#L198-L226)) |
| Curriculum stage values             | hard-coded ms targets                | [`curriculum_train.sh`](../RotaryInvertedPendulum-python/src/rl/curriculum_train.sh)                                                                                            |

## Editing the ranges

When real hardware tells you something the sim missed:

1. **Re-run sysid first.** If the nominal parameters drifted (new
   bearings, motor swap, etc.), update those before widening DR — DR
   doesn't replace good identification, it just brackets its residuals.
   See [`sysid_runbook.md`](sysid_runbook.md).
2. **Widen the matching DR range** in `pendulum_env.py`. Keep ranges
   conservative — they should bracket measured reality with margin, not
   include physically implausible regimes (those just slow training and
   teach the policy nothing useful).
3. **Re-train from scratch through the full curriculum.** Resume from
   stage-3 checkpoints is unsafe when the underlying distribution
   shifts; stage-1 will adapt the basics fastest.
4. **Validate with `eval_randomized.py`** to spot-check the policy's
   robustness across the new range before deploying.

## Related

- [`control_rate_selection.md`](control_rate_selection.md) — why dt
  jitter is the load-bearing DR knob on this rig.
- [`rl_transitions.md`](rl_transitions.md) — the `(s, a, r, s')`
  contract DR perturbs.
- [`sysid_runbook.md`](sysid_runbook.md) — measurement procedure for
  the nominal values DR brackets.
- [`async_control_architecture.md`](async_control_architecture.md) —
  how the deployment runtime preserves the timing assumptions DR
  trained against.
