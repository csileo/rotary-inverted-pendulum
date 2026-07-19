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
