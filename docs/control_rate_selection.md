# Choosing the Control Rate (and `max_action_delta_rad`)

Picking these is **not** a free choice. Done badly, the policy "works"
in sim and is broken on hardware, or vice versa. Done correctly, the
choice falls out of the sysid measurements.

This doc covers the *how* (numerical recipe) and the *why* (physical
constraints). For *what one transition looks like* once these are
fixed, see [`rl_transitions.md`](rl_transitions.md). For *how* the chosen
rate is enforced at runtime, see [`async_control_architecture.md`](async_control_architecture.md).

## Knob taxonomy: auto-derived vs user-set

The RL pipeline has many tunable parameters. They split cleanly into two
categories, and that split is enforced in the code:

**Auto-derived from other inputs** (you don't set these directly):

| Knob | Derived from | Why auto |
|---|---|---|
| `vel_filter_cutoff_hz` | `control_freq_hz` | Pure hygiene — keep filter below Nyquist + above signal bw. No design choice involved. |
| Curriculum DR delay ranges (steps) | `control_freq_hz` × physical-ms ranges | Physical delay (ms) is the meaningful unit; step count is just bookkeeping. |
| `n_substeps` (sim) | `control_freq_hz` / physics_dt | MuJoCo bookkeeping, no design content. |

**User-set** (you decide these explicitly):

| Knob | What it encodes | Why explicit |
|---|---|---|
| `control_freq_hz` | The rate window choice from sysid | Design decision: where in `[5×f_n, 3×BW_motor]` to land |
| `max_action_delta_rad` | Slew budget per tick | Design decision: how reactive should the policy be? |
| `dt_jitter_frac` (per stage) | DR strength on timing | Curriculum decision: how aggressive should regularization be? |
| `k_action`, `k_θ`, `k_motor_pos`, etc. | Policy-shaping weights | Design decision: what behavior does the reward favor? |
| Stage delay ranges (in ms, before scaling) | DR width per stage | Curriculum decision: how hard is each stage? |

**Rule**: auto-derive when there's an obvious right answer that follows
mechanically from another input. Stay explicit when the choice encodes
*intent* about how the policy should behave. Auto-derivation is hygiene;
user-set knobs are design.

This split means:
- New users on this rig don't need to remember the filter formula —
  pick a control rate and the rest of the hygiene works out.
- Override paths exist for every auto-derived knob (just pass an
  explicit value), so no magic prevents experimentation.
- The user-set knobs are the actual design surface — the small set of
  numbers where you have to think.

## Inputs (from sysid)

Two physical numbers, both produced by Phase 0 sysid:

| Quantity | How it's measured | This rig's value |
|---|---|---|
| **Motor bandwidth** `BW_motor` | `1 / (2π × rise_time_95)` from step response | **16 Hz** (64 ms rise) |
| **Pendulum natural frequency** `f_n` | `1 / period_s` from free-swing fit | **1.9 Hz** (0.526 s period) |

Both live in `sysid_params.json` after the runbook protocol. Re-derive
them whenever the rig changes (new bearings, motor swap, etc.). See
[`sysid_runbook.md`](sysid_runbook.md) for the measurement procedure.

## The valid control-rate window

The sample rate has to satisfy *both* of these inequalities simultaneously:

1. **Lower bound — pendulum-driven**: sample fast enough to control the
   unstable upright mode. Rule: `f_ctrl ≥ 5–10 × f_n`. For this rig,
   that's ≥ 10–20 Hz minimum.
2. **Upper bound — motor-driven**: don't sample faster than the motor
   can physically respond. Rule: `f_ctrl ≤ 2–3 × BW_motor`. For this
   rig, that's ≤ 32–50 Hz.

Yours: **window is roughly 30–50 Hz.** Outside it:

- Below 30 Hz: pendulum can swing significantly between ticks; closed-
  loop control becomes underdamped or unstable.
- Above 50 Hz: motor low-passes incoming targets; the policy fights
  against its own un-tracked past commands. Empirically observed: at
  100 Hz training-and-deployment, our policy's deployed avg upright
  proxy was 0.69. At 35 Hz: 0.89. Same model architecture; same hardware.
  The mismatch alone explained the gap.

## Slew rate — the per-second budget

The *slew rate* is how fast the policy is allowed to change its motor
target, measured in rad/s. It's the product of two knobs:

```
slew = max_action_delta_rad × control_freq_hz   (rad/s)
```

**What it means intuitively**: if your policy outputs the maximum action
(`a = +1.0`) every tick for one second, this is how far the motor target
moves in that second. It's the maximum demand the policy can place on
the actuator over one second of sustained extreme commands.

**Sane bound**: `slew ≤ BW_motor × A_max`, where `A_max ≈ 0.2 rad` is
the largest single setpoint step a stepper can usefully execute without
the trapezoidal profile dominating. For this rig:

`slew ≤ 16 Hz × 0.2 rad = 3.2 rad/s` (conservative)

Up to ~5 rad/s is fine in practice (motor-bandwidth headroom and the
1st-order lag don't fall off cliff-edge-suddenly).

## How it composes — this rig's numbers

| Setup | rate | delta | slew | In window? | Empirical result |
|---|---|---|---|---|---|
| Sim default (Phase 1) | 100 Hz | 0.10 | 10 rad/s | rate too high, slew too high | 0.69 upright; chattery, motor fights itself |
| 50 Hz | 50 Hz | 0.10 | 5.0 rad/s | rate at upper edge | 0.74 upright; "active correction" attractor |
| 40 Hz | 40 Hz | 0.10 | 4.0 rad/s | within window | 0.72 upright; same active attractor |
| **35 Hz** | **35 Hz** | **0.10** | **3.5 rad/s** | **calm-attractor side** | **0.91 upright; "minimal action" attractor — motor barely moves when balanced** |

## The two attractors — empirical observation

SAC reliably converges to one of two qualitatively different policies on
this rig, depending on the slew budget:

- **Calm attractor (slew ≤ ~3.5 rad/s)**: motor commands stay within
  ±0.05 rad even at full balance. Per-step action cost approaches zero.
  Visually: pendulum is statue-still, motor is essentially at rest.
  Robust against bearing noise; the policy sees the rig as
  "self-stable enough" near upright.
- **Active-correction attractor (slew ≥ ~4.0 rad/s)**: motor commands
  routinely swing ±0.5 rad even when balanced. Per-step *theta* cost
  is fine (the policy keeps θ near zero), but per-step *action* cost
  is high. Visually: pendulum jitters as motor saws back and forth.

The boundary between them is sharp — **between 35 and 40 Hz on this
rig**, with `max_action_delta_rad=0.10`. Confirmed by training a 40 Hz
policy starting from the calm 35 Hz checkpoint: 50 episodes of
fine-tuning at the higher rate flipped it into the active attractor.

The reward weights (`k_action=0.05`, `k_θ=1.0`) under-penalize action
effort relative to θ deviation. SAC will choose active correction
whenever it has the slew budget for it; the only thing keeping the 35 Hz
policy in the calm attractor is that 3.5 rad/s isn't *enough* slew for
active correction to actually beat passive stabilization in reward.

## Recipe for picking rate + delta on a new rig

## Recipe for picking rate + delta on a new rig

1. Run sysid (see [`sysid_runbook.md`](sysid_runbook.md)). Note `BW_motor`
   and `f_n`.
2. Compute the rate window: `[5 × f_n, 3 × BW_motor]`. If empty, you
   need a faster motor before this rig is controllable.
3. Pick a rate in the window. **Bias toward the lower edge.**
   Empirically (see the attractor analysis above), SAC tends to land in
   a "calm" minimal-action attractor when slew is at or below
   `BW_motor × 0.2`, and a "twitchy" active-correction attractor above
   that. Lower-edge rates also have margin against motor-BW variation
   under load. The naive "higher = more reactive" intuition fails here
   because the policy's reactivity comes from its action distribution,
   not its decision rate — and the active attractor produces *less*
   useful reactivity (just self-fighting).
4. Pick `max_action_delta_rad` so `delta × rate ≤ BW_motor × 0.2 ≈
   ~3.2-3.5 rad/s` (stays in the calm attractor on this rig).
5. Configure `control_freq_hz` and `max_action_delta_rad` consistently
   across **sim training**, **fine-tuning**, **and deployment**. The
   policy learns the rate it was trained at; mismatched deployment
   was the bug we built `async_control.py` to prevent — see
   [`async_control_architecture.md`](async_control_architecture.md).

## Sequence summary

```
sysid → BW_motor + f_n
      → rate window [5·f_n, 3·BW_motor]
      → pick f_ctrl in window (lower = conservative, higher = reactive)
      → pick max_action_delta_rad such that delta × f_ctrl ≤ ~3.5 rad/s
      → set those values once; use everywhere (sim, fine-tune, deploy)
```

## Velocity filter cutoff — picking it from the rate

The observation pipeline finite-differences raw encoder positions and
runs the result through a 1st-order low-pass IIR filter. Cutoff
frequency `vel_filter_cutoff_hz` should be picked relative to the
chosen control rate, not absolutely. If you change `control_freq_hz`,
re-derive the cutoff.

### The bound

Two inequalities, similar in spirit to the control-rate window:

1. **Lower bound — preserve the signal.** Cutoff must sit *above* the
   highest meaningful frequency in your closed-loop dynamics. For an
   inverted pendulum that's the upright instability time constant
   `1/τ = ω_n ≈ 12 Hz` for this rig. Below ~12 Hz cutoff, the filter
   attenuates the very signal the policy needs to react to.
2. **Upper bound — Nyquist.** A 1st-order filter's roll-off is gentle.
   Cutoff at or above `0.5 × control_freq_hz` (Nyquist) means each new
   sample contributes ≥75% to the running estimate — the filter is
   essentially a passthrough and isn't doing useful smoothing.

The sweet spot: **cutoff between the highest signal frequency and
~half of Nyquist**.

### This rig's numbers

Pendulum's upright-instability mode at ~12 Hz, so cutoff ≥ ~10 Hz to
preserve signal. Nyquist at the chosen rate dictates the upper bound.
The encoder noise spectrum peters out somewhere around ~20 Hz, so a
cap there avoids near-passthrough at high control rates.

### Auto-derivation (current default)

Both `real_env.py` and `run_policy.py` auto-derive the cutoff from
`control_freq_hz` if you don't pass `--vel-filter-cutoff-hz`:

```
cutoff = min(20.0, max(10.0, 0.4 × control_freq_hz))
```

| Control rate | Auto cutoff | Why |
|---|---|---|
| ≥ 50 Hz | 20.0 (capped) | Above this we'd be near-passthrough; capping at 20 keeps real filtering for noise |
| 40 Hz | 16.0 | 40% of rate, ~80% of Nyquist |
| 35 Hz | 14.0 | 40% of rate |
| **30 Hz** | **12.0** | 40% of rate, just above the 12 Hz signal edge |
| ≤ 25 Hz | 10.0 (floor) | Floor protects signal preservation; below this the filter starts attenuating useful pendulum dynamics |

The 0.4 multiplier gives ~80% of Nyquist when uncapped — solidly
filtering, not passthrough. The 10/20 Hz clamps reflect *this rig's*
signal bandwidth (~12 Hz) and noise spectrum (~20 Hz). On a different
rig these would shift with the new sysid.

### When to override

`--vel-filter-cutoff-hz N` forces a specific value. Reasons to override:

- **You see noisy actions in deployment**: cutoff might be too high.
  Drop ~2-3 Hz below the auto value.
- **You see sluggish reaction to disturbances**: cutoff too low.
  Raise ~2-3 Hz above auto.
- **You're swapping the pendulum / motor**: re-derive the rig's signal
  bandwidth and noise spectrum from sysid; pick new clamp values
  before relying on auto.

### When to retune

Change the cutoff if any of:

- **Control rate changes substantially** (more than ~2×): Nyquist
  shifts; pick a new cutoff in the new sweet spot.
- **Rig dynamics change** (different pendulum, different motor): the
  upright instability frequency shifts. Re-measure via sysid; recompute
  the lower bound.
- **You see noisy actions in deployment**: cutoff too high, dropping
  cutoff to ~12 Hz adds smoothing.
- **You see sluggish reaction to disturbances**: cutoff too low,
  attenuating useful signal. Bump up to 18-20 Hz.

## Where the chosen values live in code

| Knob | Default | Where it's set / read |
|---|---|---|
| `control_freq_hz` | **35** everywhere — `pendulum_env.py`, `real_env.py`, `async_control.py`, and `--control-freq` defaults in `train_sac.py` / `finetune_async.py` / `eval_randomized.py` / `run_policy.py` / `distill.py`. The historic 100 Hz default predated the empirical 35 Hz finding (Phase 4.6) and was wrong for this rig. | Sim env `__init__`, real env `__init__`, deployment loop |
| `max_action_delta_rad` | 0.10 in env `__init__`; `--max-action-delta-rad` flag in `train_sac.py` and `finetune_async.py` | Inside `step()` / `apply_action()` clipping |
| Curriculum stage delays (steps) | Computed in `curriculum_train.sh` from physical milliseconds × `CONTROL_FREQ` | Bash arithmetic at script top |
