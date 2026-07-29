# Rotary Inverted Pendulum — Demo

The minimal slice of this project: flash the firmware, install a small Python env, and run one of three reference policies on the physical rig — plus enough of the simulator to preview them without hardware. No training.

## 1. Flash the Arduino Nano

```bash
arduino-cli lib install FastAccelStepper AS5600
arduino-cli compile --upload -p <PORT> --fqbn arduino:avr:nano:cpu=atmega328 RotaryInvertedPendulum-arduino/LowLevelServer
```

`<PORT>` — find it with `arduino-cli board list`.

## 2. Install the Python environment

```bash
python -m venv .venv
.venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run a policy

Three reference checkpoints ship in `models/`, each trained with a different number of stacked observation frames — `--frame-stack` **must** match the value a given checkpoint was trained with, or `run_policy.py` fails to load it (`SAC.load(..., env=env)` raises on an observation-shape mismatch).

| Model | Command | What to expect |
|---|---|---|
| `policy_working_balance.zip` | `python run_policy.py --policy models/policy_working_balance.zip --frame-stack 3 --duration-s 60 --port <PORT>` | Best checkpoint so far — ~97.5% balance over 60s across two hardware validation runs on 2026-07-11, ahead of a soldering rework on this rig. Treat it as a strong starting point rather than a guaranteed result until it's re-validated on your build. |
| `policy_partial_balance.zip` | `python run_policy.py --policy models/policy_partial_balance.zip --frame-stack 1 --duration-s 15 --port <PORT>` | Works partially — swings up and balances briefly, then falls, with noticeable chattering. |
| `policy_fails_no_swingup.zip` | `python run_policy.py --policy models/policy_fails_no_swingup.zip --frame-stack 1 --duration-s 15 --port <PORT>` | Doesn't work — the pendulum won't even swing up (trained with an inflated `MAX_VELOCITY_RAD_S`, giving the policy motor authority it doesn't actually have on this rig). |
| `distill_working_balance_h32_dagger/student.pt` | `python run_policy.py --policy models/distill_working_balance_h32_dagger/student.pt --frame-stack 3 --duration-s 60 --port <PORT>` | A tiny distilled MLP (18→32→32→1) standing in for `policy_working_balance.zip` — same balance quality (upright ≈ 0.99 over multi-minute runs), small enough to eventually run standalone on the Nano itself. `run_policy.py` still does the inference on this machine either way; this is not on-device deployment. |
| `distill_working_balance_h32_dagger/student_numpy.npz` | `python run_policy.py --policy models/distill_working_balance_h32_dagger/student_numpy.npz --frame-stack 3 --duration-s 60 --port <PORT>` | Bit-exact same model as `student.pt`, re-exported to plain NumPy (`numpy_student.py`) — never imports torch/stable-baselines3. Useful wherever import time matters (e.g. a Raspberry Pi 3B+); this is what `run_demo.py` (step 5) uses by default. |

`--frame-stack 3` is `run_policy.py`'s default, so it can be omitted for `policy_working_balance.zip` and the distilled student — it's spelled out above for clarity. Ctrl-C disengages the motor cleanly in all cases.

If a checkpoint doesn't balance well on your rig, the full repo (`main` branch) ships a matching replay buffer for `policy_working_balance.zip` plus `finetune_async.py`, so you can pick up fine-tuning where it left off instead of starting from scratch.

## 4. Preview a policy in simulation (no rig needed)

No hardware, no port — this runs the checkpoint against the MuJoCo simulator instead, in a viewer window:

```bash
python train_sac.py --eval models/policy_working_balance.zip --frame-stack 3 --eval-seconds 30
```

Same `--frame-stack` caveat as above. This opens a graphical viewer, so it needs a display (no headless/SSH-only boxes).

## 5. Unattended auto-demo (e.g. a Raspberry Pi running headless)

`tools/pi_demo/run_demo.py` wraps steps 1 and 3 into a launcher meant to run
unattended on boot or on-demand: it waits for the Nano's USB cable and for
12V motor power to actually show up (in whatever order and after whatever
delay someone plugs them in), reflashes the Nano only if its firmware
doesn't already match, then runs the distilled student
(`distill_working_balance_h32_dagger/student_numpy.npz`, the torch-free
NumPy build) by default — override with `PENDULUM_POLICY`, e.g. back to
`policy_working_balance.zip`. It loads the policy and opens its Nano
connection once, then keeps reusing both across every cycle — each fresh
serial connection resets the Arduino and, unlike this default `.npz`
checkpoint, torch is slow to import on a Pi 3B+, so paying either cost
once instead of every cycle is most of what keeps cycles fast. It then
loops forever (10s pause between runs), so a headless Pi at a booth with
no network keeps re-demoing on its own — Ctrl-C or SIGTERM is the only way
to stop it. Pure Python (no udev rule, no shell script), so it runs the
same way on Linux, macOS, and Windows. See `tools/pi_demo/README.md` for
setup and the environment variables that pick a different
policy/duration/pause.
