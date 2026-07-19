# RotaryInvertedPendulum-python

Python tooling for the rotary inverted pendulum: gamepad control,
system identification, and the RL pipeline (env, SAC trainer, deployment
client). See [`../RL_PLAN.md`](../RL_PLAN.md) for the broader plan.

## Environment setup

Single mamba env named `rotary-inverted-pendulum`, Python 3.12. Conda-forge
provides the scientific stack; pip provides MuJoCo / Gymnasium / SB3 / Torch
because they are more reliably installed from PyPI.

```bash
mamba create -n rotary-inverted-pendulum -y -c conda-forge python=3.12 numpy scipy
mamba activate rotary-inverted-pendulum

# Gamepad demo (older text-protocol script)
mamba install -y pygame pyserial

# RL + sysid extras (PyPI)
pip install pyserial 'numpy<2.3' mujoco gymnasium 'stable-baselines3[extra]'
```

Notes:
- `numpy<2.3` keeps compatibility with the conda-installed `scipy 1.14`. If
  you hit `_ARRAY_API not found` errors, an older `opencv-*` in
  `~/.local/lib/python3.12/site-packages/` is shadowing the env install.
  Run `pip install --user --upgrade opencv-contrib-python` to refresh it.
- macOS only: the MuJoCo viewer requires `mjpython` (a special launcher that
  handles Cocoa main-thread quirks). Install via the same `pip` and call
  `mjpython` instead of `python` for any script that opens a viewer.

## Control rate

This rig's canonical operating rate is **35 Hz** (28.6 ms per control step,
`max_action_delta_rad = 0.10` → 3.5 rad/s slew). It's the empirically-best
operating point on this hardware: 100 Hz pushes the policy into a noisy
"active correction" attractor where the motor swings ±0.5 rad even when
balanced; 30 Hz lacks slew authority for disturbance recovery; 35 Hz lands
in the calm-attractor side of the boundary at slew ~3.5 rad/s.

All `--control-freq` flags and the `control_freq_hz` constructor defaults
across `pendulum_env.py`, `real_env.py`, `async_control.py`, `train_sac.py`,
`finetune_async.py`, `eval_randomized.py`, `run_policy.py`, and `distill.py`
default to 35 Hz. Override only if you have a reason — see
[`../docs/control_rate_selection.md`](../docs/control_rate_selection.md) for
the principled rate-window argument and the calm-vs-active attractor data.

## End-to-end pipeline

For the full sysid → sim training → real-rig fine-tune → distill → flash
walkthrough with exact commands at each step, see
[`../docs/end_to_end_runbook.md`](../docs/end_to_end_runbook.md). Each step
is idempotent, so you can also dip into individual phases without re-running
the whole pipeline.

## Workflows

### Gamepad control (legacy)

Drives the Arduino's text-protocol firmware.

```bash
python src/gamepad_control.py
```

### System identification

Phase 0 of the RL plan. See [`../docs/sysid_runbook.md`](../docs/sysid_runbook.md)
for the full protocol. Quick start (with `LowLevelServer.ino` flashed):

```bash
cd src/rl
python sysid_wizard.py                  # full pipeline: collect + fit + plots
python sysid_wizard.py fit --in-dir ... # re-fit existing logs (no rig)
```

Re-run sysid whenever the rig changes (rebuilds, new bearings, motor envelope
adjustments, etc.) so `sysid_params.json` stays current and the simulator
reflects reality.

### RL training (Phase 1+)

Trains an SB3 SAC policy against the MuJoCo env parameterised by
`sysid_params.json`. Single env on CPU is the default; vectorised envs and
GPU are available via flags.

```bash
cd src/rl
python train_sac.py --total-steps 500000 --progress-bar
# View training curves
tensorboard --logdir runs/
```

Resume / evaluate:

```bash
python train_sac.py --resume runs/<run>/last.zip --total-steps 1000000
mjpython train_sac.py --eval runs/<run>/best_model.zip --eval-seconds 30
```

### Curriculum training (recommended)

Training from scratch with the full target DR (4-7 step delay, etc.) is
hard for SAC to converge — the policy gets stuck early. The curriculum
runner trains the policy in three stages of increasing DR difficulty,
each `--resume`-ing from the previous one's `best_model.zip`.

```bash
./curriculum_train.sh <run-name-prefix>
# example -> runs/<prefix>_stage1, _stage2, _stage3
```

Stage breakdown (defined in `curriculum_train.sh`):

| Stage | tau range | delay range | steps |
|---|---|---|---|
| 1 (easy)   | 0–5 ms  | 0–2 steps | 100k |
| 2 (medium) | 0–10 ms | 2–5 steps | 100k |
| 3 (final)  | 0–10 ms | 4–7 steps | 100k |

The final stage 3 delay range matches the hardware's measured 5-step
transport delay at `MOTOR_ACCELERATION = 50k` and `Vref = 0.45 V`.
Override per-stage parameters via env vars (`SEED`, `STEPS_PER_STAGE`,
`DEVICE`).

### Evaluating a policy

Two complementary tools:

```bash
# Render a deterministic eval rollout in the MuJoCo viewer (mjpython on macOS)
mjpython train_sac.py --eval runs/<run>/best_model.zip --eval-seconds 30

# Stress-test the policy on N=20 randomly-physics-configured episodes.
# Reports a per-episode ✓/✗, a "solved" success rate, and the underlying
# DR sample (tau, delay) used for each episode. See the script docstring
# for the full success-criterion definition.
python eval_randomized.py runs/<run>/best_model.zip --n-episodes 20
```

### Trajectory logging + sim-to-real analysis

`run_policy.py --log <path>.npz` saves a step-level trajectory of the real-
hardware run. Two helper scripts then introspect that log to fit refined
sysid against the actively-driven dynamics (typically more informative
than the chirp/step sysid alone):

```bash
# Rough fit of effective transport delay and first-order tau against
# motor_target → motor_actual.
python analyze_run.py /tmp/policy_run.npz

# Replay the same logged action sequence in the sim and report motor /
# pendulum trajectory error vs the real device. Shows you how big the
# sim-to-real gap actually is on a real run.
python sim_vs_real.py /tmp/policy_run.npz
```

### Real-hardware deployment (Phase 3)

Drives the Arduino's `LowLevelServer.ino` binary protocol. Re-flash the
Arduino with `LowLevelServer` (sysid uses a different sketch). Always run
`--dry-run` first to verify the protocol; then run engaged with the rig
supervised.

```bash
cd src/rl
python run_policy.py --policy runs/<run>/best_model.zip \
    --port /dev/cu.usbserial-1130 --duration-s 5 --dry-run
python run_policy.py --policy runs/<run>/best_model.zip \
    --port /dev/cu.usbserial-1130 --duration-s 5
```

The deploy client clamps commanded position to ±125° (inside the ±135°
mechanical hard stops) and disengages the motor on exit / SIGTERM /
SIGINT.

## Layout

```
src/
  gamepad_control.py        # legacy text-protocol gamepad control
  rl/
    pendulum_env.py         # Gymnasium env (MuJoCo) parameterised from sysid_params.json
    train_sac.py            # SB3 SAC trainer with eval+checkpoint callbacks
    curriculum_train.sh     # 3-stage curriculum runner (easy -> full DR)
    eval_randomized.py      # N-episode randomized-env stress test (success-rate metric)
    run_policy.py           # Phase 3 deployment client (binary protocol)
    analyze_run.py          # Fits effective tau / transport delay from a run_policy log
    sim_vs_real.py          # Replays a logged action sequence in sim and compares
    lowlevel_client.py      # Python client for LowLevelServer.ino
    sysid_wizard.py         # Interactive sysid (collect/fit/validate-motor)
    sysid_core.py           # Sysid math (fits + parameter derivation)
    sysid_params.json       # Output of sysid_wizard.py (committed)
    freeswing_probe.py      # Standalone sim-vs-real validator
    sysid_runs/             # Per-session recordings (gitignored)
    runs/                   # Training artifacts (gitignored)
test/
  serial_test.py            # bare serial sanity check
  gamepad_test.py           # gamepad sanity check
```
