# Selected models

Three checkpoints representative of the project's progression, with their
hardware validation log (.npz files, readable via numpy.load). Not the full
history of runs/checkpoints — just one broken point, one partial point, and
the current best checkpoint.

| File | State | Associated log | Replay buffer |
|---|---|---|---|
| `policy_fails_no_swingup.zip` | Doesn't work: the pendulum won't even swing up (`MAX_VELOCITY_RAD_S` was set to 7.0 rad/s, giving the policy fictitious motor authority) | `logs/policy_fails_no_swingup.npz` | None — checkpoint from a pure-simulation training run, never fine-tuned on the real rig |
| `policy_partial_balance.zip` | Works partially: swings up and balances briefly, then falls, with noticeable chattering | `logs/policy_partial_balance.npz` | None — the only buffer available for this run corresponds to episode 30's state, not episode 71's; including it would have been misleading |
| `policy_working_balance.zip` | Best checkpoint so far (from `finetune_curriculum8_v5`): ~98.5% balance over a 60s validation run, catches upright in 0.88s and holds it for 59.1s straight (never falls) | `logs/policy_working_balance.npz` | `policy_working_balance_replay_buffer.pkl` — to resume fine-tuning with `finetune_async.py --resume-buffer` |

Superseded checkpoints (previous `policy_working_balance` and `_v4`), with their logs
and replay buffers, are kept in `experiments/` for reference.

## Pre-fine-tuning comparison point

`policy_working_balance_before_finetuning.zip` is `policy_working_balance.zip`'s
direct ancestor: the pure-simulation checkpoint (`curriculum8_stage1/best_model.zip`,
archived 04/07/2026) from *before* any real-rig fine-tuning
(`finetune_curriculum8` onward — see `PLAN/RL.md`, Étape 15). Useful to
demonstrate the sim/real gap: `policy_working_balance.zip` went through 60+
real-rig fine-tuning episodes and now fails to balance in MuJoCo simulation
(specialised away from sim dynamics), while this earlier checkpoint still
balances fine in the simulator:

```bash
python train_sac.py --eval models/policy_working_balance_before_finetuning.zip \
    --control-freq 50 --frame-stack 3 --reward-motor-jerk-weight 0.01 --eval-seconds 30
```

`--control-freq 50` is required — this checkpoint was trained at 50 Hz, not the
35 Hz `train_sac.py --eval` otherwise defaults to (see that script's own
warning comment: evaluating at the wrong rate "produces garbage").

No replay buffer (`.pkl`) ships alongside it — the archived run directory
(`experiments/runs/curriculum8_stage1/` at the `SimToReal` repo root) never
had one: `train_sac.py`'s plain simulation training doesn't save a replay
buffer the way `finetune_async.py`'s real-rig sessions always do.

## Distilled student

`distill_working_balance_h32_dagger/student.pt` is not a SAC checkpoint — it's
a tiny MLP (18 → 32 → 32 → 1, i.e. a 3-frame-stacked observation) distilled
from `policy_working_balance.zip`, small enough to eventually run standalone
on the Arduino Nano. It matches the teacher's balance quality (upright ≈
0.987-0.988 over multi-minute runs, validated 2026-07-28 both tethered to a
PC and to a Raspberry Pi via `run_policy.py --policy
models/distill_working_balance_h32_dagger/student.pt --frame-stack 3`) — no
`--log` capture was saved for those validation runs, so there's no matching
`logs/*.npz` file the way there is for the SAC checkpoints above.

The directory also ships `dataset.npz`: 301,461 `(obs, action_target)` pairs
— the original teacher-replay dataset `distill.py` builds, plus a DAgger
refresh pass (`dagger_relabel.py`) that relabels states the deployed student
actually visited (including its own near-fall recovery windows) with the
teacher's action. That dataset, not a hardware log, is what
`distill.py`/`dagger_relabel.py` need to regenerate `student.pt` from
scratch.

`student_numpy.npz` is the same weights re-exported to plain NumPy by
`export_weights_numpy.py` (bit-exact to `student.pt`, max diff ~9e-7 —
float32 rounding noise) — nothing but `numpy_student.py` is needed to run
it, so a caller (`tools/pi_demo/run_demo.py`) never has to import
torch/stable-baselines3 at all. Regenerate it after retraining with:

```bash
python export_weights_numpy.py \
    --student models/distill_working_balance_h32_dagger/student.pt \
    --out models/distill_working_balance_h32_dagger/student_numpy.npz
```

This student is validated tethered only (`run_policy.py` doing inference on
a PC or Raspberry Pi, talking to the Nano over serial as a dumb sensor/motor
server) — not a standalone on-device deployment. `docs/quantisation.md`
covers int8 quantisation for an earlier, non-frame-stacked student
architecture; it predates this DAgger-refreshed, frame-stacked one and
isn't a guide to this checkpoint's on-device prospects.
