# distill_working_balance_h32 — throwaway test artifact

Student distilled from `models/policy_working_balance.zip` (teacher) +
its real-rig replay buffer, augmented with 100k sim rollout steps,
`--control-freq 35` (matches the teacher's training rate). See the commit
history on this branch for the exact `distill.py` invocation.

- `student.pt` — 18 -> 32 -> 32 -> 1 MLP (raw obs is 6-dim, teacher was
  trained with `--frame-stack 3`, so 6*3=18 input features), 1697 params.
- `dataset.npz` — the (obs, action_target) training set, kept for
  reproducibility.
- Results: final val_mse=0.007682 (acceptance threshold ~0.02), numpy/
  PyTorch parity max diff 7.078e-07 (tol 1e-05). Not yet validated on
  hardware.

## Option A — run the student directly in Python (Pi or laptop, no flash)

```bash
cd RotaryInvertedPendulum-python/src/rl
python run_policy.py \
    --policy models/distill_working_balance_h32/student.pt \
    --port <PORT> --duration-s 30
```

`<PORT>`: find it with `arduino-cli board list`. No `--frame-stack` flag
here — `run_policy.py` reads frame-stacking straight from the student
checkpoint for `.pt` policies. Compare the printed `upright` proxy
against the teacher's ~98.5% (60s run) — a gap > 0.05 means the
distillation under-fit (try `--sim-augment-steps 200000` or a wider
`--hidden`, see `docs/end_to_end_runbook.md` step 5).

## Option B — flash the standalone Nano sketch (no Pi/laptop at runtime)

The weights are already exported to
`RotaryInvertedPendulum-arduino/RLControl/policy_weights.h` on this
branch (see the commit after this one). To flash:

```bash
cd RotaryInvertedPendulum-arduino/RLControl
arduino-cli compile --upload -p <PORT> \
    --fqbn arduino:avr:nano:cpu=atmega328 .
```

`<PORT>`: same as above. This sketch runs the policy fully on the Nano —
no serial link to a computer needed at runtime. Tether-test with Option A
first before trusting this on hardware unattended.
