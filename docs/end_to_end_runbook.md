# End-to-End Runbook

How to take a freshly-built rig from "no policy at all" to "balances
standalone on the Nano, no laptop tether". Each step lists the command,
the expected wall-clock cost, and what the next step depends on.

For *why* the pipeline is shaped this way, see [`../RL_PLAN.md`](../RL_PLAN.md).
For *how a single transition works*, see [`rl_transitions.md`](rl_transitions.md).

```
   ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │ 0 sysid  │───▶│ 1 sim    │───▶│ 2 fine-tune  │───▶│ 3 test       │
   │  recordings    curriculum     real-rig async      teacher tethered
   └──────────┘    └──────────┘    └──────────────┘    └──────┬───────┘
                                                              │ scored ≥ 0.9?
                                                              ▼
                                  ┌──────────────────────┐    ┌──────────┐
                                  │ 6 flash + verify     │◀───│ 4 distill│
                                  │ on-device standalone │    │ student  │
                                  └──────────────────────┘    └────┬─────┘
                                                                   │
                                                              ┌────▼──────┐
                                                              │ 5 test    │
                                                              │ student   │
                                                              │ tethered  │
                                                              └───────────┘
```

If you only need the upright/balance behaviour and are happy keeping the
laptop attached: stop after step 3. Steps 4–6 only exist to remove the
tether.

## Prerequisites

- macOS / Linux dev box with `arduino-cli`, the `arduino:avr` core, and
  the `AS5600` (RobTillaart) + `FastAccelStepper` libraries installed.
- Python env per [`../RotaryInvertedPendulum-python/README.md`](../RotaryInvertedPendulum-python/README.md):
  `mamba activate rotary-inverted-pendulum`.
- Rig wired with **STEP on pin 9**, DIR on pin 2, ENABLE on pin 5, AS5600
  on I²C (A4/A5). Pin 9 is required by FastAccelStepper on ATmega328 and
  works for AccelStepper too — see
  [`../NEXT_STEPS.md`](../NEXT_STEPS.md).

In the commands below, replace `/dev/cu.usbserial-1130` with whatever
port `arduino-cli board list` shows for your Nano.

## 0. System identification — measure the rig

Pinning the dynamics parameters once. Outputs `sysid_params.json`,
which `pendulum_env.py` reads to build the sim.

Full protocol: [`sysid_runbook.md`](sysid_runbook.md). Re-run any time
the rig changes mechanically (new bearings, rebuilt arm, changed
microstepping, swapped motor).

## 1. Train the teacher in sim — curriculum

The repo's canonical recipe trains a SAC actor through three DR stages
of increasing realism. End-to-end ~25 min on the MacBook (CPU, single
env). Outputs `runs/<run-name>/last.zip`.

```bash
cd RotaryInvertedPendulum-python/src/rl
bash curriculum_train.sh
```

`curriculum_train.sh` reads `sysid_params.json`, derives DR ranges from
physical-time-units, runs three stages, and writes the final policy
plus checkpoints. The `control_freq_hz` is **35 Hz** by default in
every component — see [`control_rate_selection.md`](control_rate_selection.md)
for why.

## 2. Fine-tune on the real rig — async

The sim policy will not balance on hardware on its first try
(sim-to-real gap). Async fine-tuning closes that gap with ~30–80
real-rig episodes (~10–25 min wall clock).

Flash the LowLevelServer first so the laptop can drive the rig over the
binary protocol:

```bash
arduino-cli compile --upload -p /dev/cu.usbserial-1130 \
    --fqbn arduino:avr:nano:cpu=atmega328 \
    RotaryInvertedPendulum-arduino/LowLevelServer
```

Then run the async orchestrator. `--resume-buffer` is optional on the
first session; on subsequent sessions, point it at the previous run's
`replay_buffer.pkl` to keep accumulated real-rig transitions:

```bash
cd RotaryInvertedPendulum-python/src/rl

# First fine-tune session
python finetune_async.py \
    --policy runs/<sim-run>/last.zip \
    --port /dev/cu.usbserial-1130 \
    --episodes 50 \
    --run-name async_v1

# Subsequent sessions, buffer-resumed
python finetune_async.py \
    --policy runs/async_v1/last.zip \
    --resume-buffer runs/async_v1/replay_buffer.pkl \
    --port /dev/cu.usbserial-1130 \
    --episodes 30 \
    --run-name async_v1_extend
```

Architecture detail: [`async_control_architecture.md`](async_control_architecture.md).

The orchestrator disengages the motor for `--reset-settle-s` (default 5)
between episodes so the pendulum coasts to rest passively. **Listen** to
the motor during the first few episodes — a smooth whirr is fine, a
buzzy/grinding sound means step-skipping (drop `MOTOR_ACCELERATION` in
`LowLevelServer.ino` and `RLControl.ino` from 50 k → 30 k and re-flash).

## 3. Test the teacher on the rig — tethered

Confirms the fine-tuned teacher actually balances before spending more
time on it. Cheap (30 seconds of rig time).

```bash
python run_policy.py \
    --policy runs/async_v1_extend/last.zip \
    --port /dev/cu.usbserial-1130 \
    --duration-s 30
```

Watch the printed `upright` proxy at the end:
- **mean ≥ 0.9** → solid teacher, proceed to step 4 to remove the tether.
- **mean 0.85–0.9** → acceptable; can proceed but a few more fine-tune
  episodes (back to step 2 with `--resume-buffer`) usually buys
  robustness.
- **mean < 0.85** → fine-tuning didn't converge. Diagnose before
  distilling: check the
  [policy improvement ideas](policy_improvement_ideas.md) backlog,
  consider re-sysid, consider a wider `--gradient-steps`.

If you're happy keeping the laptop attached, **you can stop here**.
The teacher runs at 35 Hz over USB serial just fine.

## 4. Distill — shrink the actor for the Nano

The fine-tuned SAC actor is 67 K parameters (≈ 270 KB float32) — far
too big for the Nano's 32 KB flash. Distill into a 5→32→32→1 student
(≈ 5 KB). Cheap (~1 minute):

```bash
python distill.py \
    --teacher runs/async_v1_extend/last.zip \
    --buffer  runs/async_v1_extend/replay_buffer.pkl \
    --out-dir runs/async_v1_extend/distill_h32_aug
```

`distill.py`:
1. Loads teacher + replay buffer (real-rig observations).
2. Re-evaluates the deterministic-mean teacher action on each obs.
3. Augments with 100 K teacher rollouts in the DR sim (helps when
   the real-rig buffer is small/sparse).
4. Trains a tiny MLP via supervised regression with a tanh head.
5. Sanity-checks numpy parity with PyTorch (catches export bugs).

Acceptance for the student: validation MSE ≲ 0.02 in action units, plus
the tethered rollout in the next step. Closed-loop sim eval was removed
from `distill.py` because it isn't a meaningful gate for real-rig
fine-tuned teachers — they routinely fail the sim eval even when they
balance perfectly on hardware.

## 5. Test the student on the rig — tethered

Confirms the student is a faithful distillation of the teacher *before*
flashing it onto the Nano. Same `run_policy.py` flow but with the `.pt`
file. Cheap (30 s of rig time):

```bash
python run_policy.py \
    --policy runs/async_v1_extend/distill_h32_aug/student.pt \
    --port /dev/cu.usbserial-1130 \
    --duration-s 30
```

Expect the upright proxy to be **within ~0.05** of the teacher's score
from step 3. Larger gap → covariate-shift problem; either grow the sim
augmentation (`distill.py --sim-augment-steps 200000`), increase student
capacity (H=48 still fits comfortably), or accept and move on.

## 6. Flash the standalone sketch — remove the tether

```bash
# Export PROGMEM weights into the Arduino sketch directory
python export_weights.py \
    --student runs/async_v1_extend/distill_h32_aug/student.pt \
    --header  ../../../RotaryInvertedPendulum-arduino/RLControl/policy_weights.h \
    --source-name async_v1_extend/distill_h32_aug

# (Optional but recommended) update the boot self-test reference values
# in RLControl.ino to match the new student. Compute via:
#   python -c "import torch, numpy as np; from distill import StudentMLP, _student_predict_factory; \
#       ckpt = torch.load('runs/async_v1_extend/distill_h32_aug/student.pt', \
#           map_location='cpu', weights_only=True); \
#       m = StudentMLP(hidden=ckpt['hidden'], obs_dim=ckpt['obs_dim'], act_dim=ckpt['act_dim']); \
#       m.load_state_dict(ckpt['state_dict']); pred = _student_predict_factory(m); \
#       print('hanging:', float(pred(np.array([0,0,-1,0,0],dtype=np.float32))[0])); \
#       print('upright:', float(pred(np.array([0,0,1,0,0],dtype=np.float32))[0]))"

# Flash
cd ../../..
arduino-cli compile --upload -p /dev/cu.usbserial-1130 \
    --fqbn arduino:avr:nano:cpu=atmega328 \
    RotaryInvertedPendulum-arduino/RLControl
```

Pin the pendulum hanging straight down at the moment the **3 s startup
delay** ends (when the LED switches from slow blink to fast blink) —
that pose becomes the encoder zero for the engagement.

Verify on serial monitor at 500 kbaud:
- Boot prints `[boot] policy(hanging) = X.XXXX` and
  `[boot] policy(upright) = X.XXXX`. These should match the values you
  computed in the helper one-liner above to ≤ 1e-3 (AVR float vs
  PyTorch). If they don't, the C++ forward pass / PROGMEM access is
  broken — re-export, recompile, re-flash.
- Once engaged: pendulum should swing up and balance within ~3 s, hold
  ≥ 30 s undisturbed, recover from a flick disturbance.

## Re-running individual steps

Every step is idempotent and can be re-run on its own:

| Want to | Re-run | Resume from |
|---|---|---|
| Tweak rewards or DR ranges | step 1 | scratch |
| Add real-rig data | step 2 | `--resume-buffer` |
| Try a smaller / larger student | step 4 | existing teacher |
| Re-flash with the same student | step 6 | existing `.h` |

## Troubleshooting

- **Teacher balances tethered but student fails on Nano**: the most
  likely bug is the encoder zero — captured *at engage time* in
  `RLControl.ino`, but if the sketch was uploaded with the pendulum in
  a non-hanging pose and not re-positioned during the 3 s delay, the
  policy frame is rotated. Reset the Arduino with the pendulum hanging.
- **Boot self-test prints `[FATAL] FastAccelStepper config rejected`**:
  the requested `MOTOR_MAX_SPEED` exceeds FastAccelStepper's AVR cap of
  50 kSteps/s for a single stepper. Check the constant in
  `RLControl.ino`.
- **Pendulum swings but never reaches upright**: motor authority is
  too low. Verify the `MOTOR_ACCELERATION` matches between
  `LowLevelServer.ino` (used during fine-tuning) and `RLControl.ino`
  (used at deployment). They must agree, otherwise the policy is
  trained against one set of dynamics and deployed against another.
