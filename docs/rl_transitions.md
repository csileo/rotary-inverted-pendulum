# RL Transitions, in Plain English

What one `(s, a, r, s')` tuple in this project actually contains, end to
end. Reference both for sim (`pendulum_env.py`) and real-device
(`real_env.py`) environments — they're identical by design so a
sim-trained checkpoint can keep learning on the real rig without any
adapter.

## The setup in one paragraph

A rotary inverted pendulum (Furuta type). A horizontal arm rotates in
the floor plane, driven by a stepper motor (the "motor" angle). At the
end of that arm hangs a free-swinging pendulum (the "pendulum" angle).
The goal is to swing the pendulum from hanging-down to upright and hold
it there. Every 10 ms the policy sees the current state, picks a
control nudge, and the world moves on.

## Conventions and frame

- **Pendulum angle θ**: 0 means **upright**, ±π means hanging-down.
  We use θ rather than the raw MuJoCo joint angle φ everywhere in the
  observation and reward, because it makes "the goal" trivially `θ=0`.
  Internally `θ = wrap_pi(φ − π)`.
- **Motor angle**: 0 is the calibrated centre at startup. Positive =
  counter-clockwise looking down. Mechanical hard stops at ±135°; we
  clamp commanded targets to ±125° so the policy never asks for a stop
  hit.
- **Control rate**: configurable (`control_freq_hz`). Sim physics
  integrates at 1 kHz under the hood; real env paces wall-clock to
  match. The choice of rate is rig-specific and bounded by motor
  bandwidth and pendulum dynamics — see
  [`control_rate_selection.md`](control_rate_selection.md). The runtime
  enforcement of whatever rate is chosen is described in
  [`async_control_architecture.md`](async_control_architecture.md).

## Observation `s` — what the policy sees (5 floats)

```
s = [motor_pos, sin(θ), cos(θ), motor_vel, pendulum_vel]
```

| Component | Units | Range | Meaning |
|---|---|---|---|
| `motor_pos` | rad | ±2.36 (= ±135°) | Current arm angle from centre |
| `sin(θ)` | — | [−1, 1] | Sine of pendulum-from-upright |
| `cos(θ)` | — | [−1, 1] | Cosine of pendulum-from-upright. **= 1 at the goal**, = −1 hanging down |
| `motor_vel` | rad/s | ±200 | Arm angular velocity |
| `pendulum_vel` | rad/s | ±200 | Pendulum angular velocity |

Why sine + cosine instead of θ directly? θ wraps at ±π, so the policy
network would see a discontinuity right next to one of its operating
points (hanging down). `(sin θ, cos θ)` is a continuous unit-circle
embedding — no jump, no gradient cliff.

**How `s` is built**:

- **Sim** (`pendulum_env.py::_obs`): read `qpos`/`qvel` directly from
  MuJoCo. With domain randomisation on, we additionally quantise the
  pendulum angle to AS5600 LSB (12-bit, ~0.0015 rad) and inject small
  Gaussian noise on positions and velocities — so the sim observation
  pipeline matches what the real rig actually delivers.
- **Real** (`real_env.py::_build_obs`): poll the LowLevelServer over
  serial → un-flip the firmware's sign convention → finite-difference
  the velocities and run them through a 20 Hz low-pass filter. The
  filter is critical: raw finite-difference at 100 Hz on a noisy
  encoder is unusable.

## Action `a` — what the policy outputs (1 float)

```
a ∈ [−1, 1]
```

The action is **not a torque or a position**. It's a normalised *delta*
applied to the motor's commanded target each step:

```
motor_target ← clip(motor_target + a · max_action_delta_rad, ±125°)
```

So the policy steers the motor by issuing per-step nudges. The `clip`
enforces the soft motor limit so the policy literally cannot command a
hard-stop hit, even if it wants to.

`max_action_delta_rad` is one of two coupled knobs (the other is
`control_freq_hz`). Their product is the *slew rate* in rad/s, which
must respect the motor's bandwidth — see
[`control_rate_selection.md`](control_rate_selection.md) for the
rationale and recipe.

This action representation has two key benefits:

1. The stepper firmware (AccelStepper) accepts position targets, not
   torques. Mapping action directly to a position-delta matches the
   hardware interface.
2. Smooth-by-construction: between consecutive steps the commanded
   position can change by at most 0.1 rad, regardless of policy
   craziness. This bounds the worst-case actuator slew rate and
   protects the motor from policy exploration during training.

## Transition dynamics — going from `s` to `s'`

In **sim** (per step):

1. Apply the action delay queue (DR samples a 0–N-step delay scaled to
   bracket the rig's measured transport delay; the action that takes
   effect now might be the one the policy chose several steps ago —
   modelling serial RTT + AccelStepper ramp).
2. Update the commanded `motor_target` with the (delayed) action.
3. Apply motor first-order lag (DR samples τ ∈ [0, 10] ms): the
   `motor_applied` value fed to MuJoCo trails the commanded target by
   an exponential of time-constant τ. With τ=0 it's instantaneous.
4. Step MuJoCo physics for one control period at 1 ms substeps.
5. Read out the new state, build `s'`, compute reward.
6. Episode terminates if the motor hits the hard stop at ±135° (incurs
   a `−5` penalty); truncates after `episode_length_s` (default 8 s).

In **real** (per step):

1. Update the commanded `motor_target` with the action (no action delay
   queue — the real hardware *is* the delay).
2. Send `set_target(motor_target)` over serial.
3. Sleep until the next tick (paces wall-clock to control rate).
4. Read state from the rig: `(time_us, motor_pos, pendulum_pos)`.
5. Finite-diff + low-pass filter the velocities.
6. Build `s'`, compute reward.
7. Terminate on |motor_pos| ≥ 135° (firmware also enforces this in
   hardware as a backstop); truncate after `episode_length_s` (default
   6 s during fine-tuning).

The key sim-to-real bridge is that **the firmware's transport delay,
acceleration ramp, and stepper friction are what the sim's
`action_delay_steps`, `motor_tau_s`, and joint friction parameters are
modelling**. Domain randomisation samples those over plausible ranges
each episode so the policy has seen enough variation to handle the
real point.

## Reward `r` — what we're rewarding

The current reward is the **standard Quanser quadratic-cost form**
(common in Furuta-pendulum literature):

```
r = −[ θ² + k_θ̇·θ̇² + k_α·α² + k_α̇·α̇² + k_a·a² ]
```

where:

- **θ** = pendulum-from-upright (rad). The dominant term: `θ²` is 0
  at the goal and ≈ π² ≈ 9.87 at hanging-down.
- **θ̇** = pendulum_vel. Small `k_θ̇=0.001` weight discourages
  spinning through upright forever.
- **α** = motor_pos (rad, sim's misnamed-from-Quanser variable).
  `k_α=0.5` keeps the policy near centre.
- **α̇** = motor_vel. `k_α̇=0.005` discourages frantic arm motion.
- **a** = action ∈ [−1, 1]. `k_a=0.05` light penalty for jerky control.

Reward is **purely non-positive** — max 0 when fully balanced still at
centre with no motor activity, around −10 per step at hanging-down.
SAC handles negative rewards fine, and the all-negative signal makes
"less negative" gradient toward upright unambiguous.

**What the policy actually learns to do:**

- *Far from upright*: Pump the arm back and forth. The `θ²` term
  rewards getting upright; the `α²` and `α̇²` penalties keep the
  pumping bounded so it doesn't slam into the limits.
- *Near upright*: Hold still. The dominant `θ²` term goes near zero
  there, leaving only the small velocity/action penalties as residuals
  — so the policy is rewarded for any state close to (θ=0, θ̇=0,
  α=0, α̇=0).

If the policy gets the pendulum near upright but not still, the cost
is dominated by `k_θ̇·θ̇²`. If it balances but with the arm wandering,
the cost is dominated by `k_α·α²`. These weights are what shape the
policy from "wobbly catch" toward "smooth hold".

## Episode boundaries

| Event | What happens | When |
|---|---|---|
| **Reset** | Sim places pendulum hanging-down with small noise, motor at random `±0.7·motor_safe_limit`. Real disengages motor, waits `reset_settle_s` for pendulum to coast to rest, re-engages at current motor position. | Every episode |
| **Termination** | `terminated = True`, reward gets a final `−5` penalty. Episode boundary. | Hard-stop hit (`|motor_pos|≥135°`) |
| **Truncation** | `truncated = True`, reward unaffected. Episode boundary. | Time limit reached (8 s sim, 6 s real default) |

Truncation just bookkeeps the time limit — the value-of-future is still
estimated normally. Termination signals the value should drop to zero
("game over") and is reserved for the hard-stop bad outcome.

## Why this representation works

A few non-obvious choices, called out:

- **`(sin θ, cos θ)` over θ**: avoids the wraparound discontinuity. The
  policy sees a smooth manifold, not a step function.
- **Action as delta-in-target, not absolute target**: by integrating
  the policy output, we let the policy *steer* rather than *hop*. Slew
  rate is bounded; motor can never be commanded to teleport.
- **Reward purely negative**: simpler optimisation surface than mixed
  positive/negative. SAC's entropy bonus handles exploration; we don't
  need positive shaping bonuses.
- **Same env for sim and real**: zero translation cost on
  checkpoint-load. The replay buffer in Phase 4 fine-tuning fills with
  real (s, a, r, s') tuples that look identical in shape to the sim
  ones the policy already learned from.

## Where things live in code

| File | What it does |
|---|---|
| `pendulum_env.py` | The full sim env — MJCF model, DR, action delay, motor lag, reward. The canonical reference. |
| `real_env.py` | Hardware version. Deliberately mirrors `pendulum_env.py`'s observation, action, and reward exactly. |
| `run_policy.py` | Deployment-only client. Same observation pipeline as `real_env.py`, no learning. |
| `async_control.py`, `finetune_async.py` | Runtime that *produces* transitions during fine-tuning at strict rate. Internals out of scope here — see [`async_control_architecture.md`](async_control_architecture.md). |
| `finetune_real.py` | Deprecation shim → forwards to `finetune_async.main`. |

Read `pendulum_env.py::step` and `pendulum_env.py::_obs` together for
the canonical sim transition; read `real_env.py::step` and
`real_env.py::_build_obs` to see the same flow against hardware.

## See also

- [`async_control_architecture.md`](async_control_architecture.md) — how
  the rig's control loop is held to a strict rate during fine-tuning,
  decoupled from SAC's gradient updates.
- [`control_rate_selection.md`](control_rate_selection.md) — how to
  pick `control_freq_hz` and `max_action_delta_rad` from sysid
  measurements (motor bandwidth + pendulum natural frequency).
- [`sysid_runbook.md`](sysid_runbook.md) — the measurement procedure
  that produces the inputs both of those docs depend on.
