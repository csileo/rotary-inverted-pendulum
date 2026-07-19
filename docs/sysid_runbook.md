# System Identification Runbook

End-to-end protocol for measuring the physical parameters of your
pendulum and writing a `sysid_params.json` that the RL pipeline consumes.

Sysid runs against the **same** `LowLevelServer.ino` firmware the policy
deploys against — no firmware swap, no risk of measurement-vs-deployment
parameter drift.

The wizard is split into two phases:

- **collect** — operator-driven recording on the rig. Writes raw `.npz`
  logs + a `metadata.json` to a timestamped directory.
- **fit** — pure post-processing. Reads a directory of recordings, derives
  parameters, writes `sysid_params.json`, and renders sim-vs-real
  validation plots. No device required; re-runnable as often as you like.

Re-fitting from saved logs is the fast iteration loop when improving the
math. Re-collecting on the rig is only needed when changing sensor read
rates, firmware logic, or adding new recording steps.

## Prerequisites

- Python env: `mamba activate rotary-inverted-pendulum` with
  `numpy`, `scipy`, `pyserial`, `matplotlib`, `mujoco` installed.
- `LowLevelServer.ino` flashed onto the Arduino Nano. Auto-magnet-detect
  at boot; the firmware halts in `setup()` until it sees the pendulum
  magnet.
- A small rigid block (cardboard, foam, wood) to slide under the pendulum
  for the free-swing release.
- Pendulum geometry (mass, COM, inertia) comes from `urdf/model.urdf` —
  no operator-measured inputs are needed. If you've rebuilt or modified
  the pendulum, update Onshape → export → URDF first.

## Recommended path: full pipeline

```bash
cd RotaryInvertedPendulum-python/src/rl
python sysid_wizard.py
```

Walks you through, in order:

1. **Tare hanging position.** Pendulum hangs motionless; we sample for
   3 s and use the median as "physical hanging = 0" in the saved
   coordinate frame. Fixes the firmware-encoder-accumulator drift.
2. **Free-swing × 3.** Hold the motor arm firmly, slide the block under
   the pendulum to lift it (any reasonable angle — the sensor records the
   actual release angle), then slide the block out perpendicular to the
   swing plane. Recording is 10 s by default. Repeat three times.
3. **Motor ±90° sanity sweep.** Engages the motor and drives it through
   `0 → +90° → 0 → −90° → 0` using accel commands. Watch the rig: the
   arm should reach roughly perpendicular at the peaks. If it visibly
   under-shoots, the stepper is skipping steps and the firmware
   position-counter no longer matches the actual mechanical position
   — a deployment-blocker (lower `MOTOR_ACCELERATION` or bump the
   DRV8825 Vref).
4. **Fit + plots.** Aggregates the three free-swing recordings, derives
   friction parameters using URDF-defined geometry, writes
   `sysid_params.json`, generates `freeswing_compare.png` (sim trace
   overlaid on the real recording using the just-derived params) and
   `motor_sweep.png` (motor target vs. actual during the sanity sweep).

Pendulum mass, COM distance, and inertia about the COM are *not* part
of this pipeline anymore — they're geometric properties of the part and
live in `urdf/model.urdf`, parsed by `pendulum_geometry.py`. The free-
swing fit cross-checks the URDF inertia against the measured period
(a >10% mismatch flags a likely-stale URDF).

The whole thing takes ~10–15 minutes. Output ends up in
`sysid_runs/<timestamp>/` next to the wizard.

## Re-fit from existing recordings (no rig needed)

```bash
python sysid_wizard.py fit --in-dir sysid_runs/2026-05-20_090000
```

Use this when iterating on `sysid_core.derive_pendulum_friction` or
investigating the fits — the rig stays unused, and the operator doesn't
need to be present.

## Standalone motor sanity sweep

```bash
python sysid_wizard.py validate-motor
```

Just the motor ±90° sweep + plot. Useful after any change to the
firmware's motor configuration (current limit, microstepping,
acceleration ceiling) to confirm step counting still tracks reality.

## What the wizard writes

```
sysid_runs/2026-05-20_HHMMSS/
├── metadata.json               # port, firmware, timestamp, ...
├── tare.npz                    # pendulum-at-rest recording → hanging zero
├── free_run_1.npz              # raw free-swing recordings
├── free_run_2.npz
├── free_run_3.npz
├── motor_sweep.npz             # motor target/actual during sanity sweep
├── freeswing_compare.png       # sim-vs-real overlay for validation
└── motor_sweep.png             # motor target vs. firmware-reported position
```

`sysid_params.json` (the only file the RL pipeline reads) is written at
the project root by default.

## What the math does

`fit_free_swing(t, θ)` (in `sysid_core.py`) finds peaks, computes the
period and damped-oscillator decay constants. It also computes the
**small-amplitude period** T₀ via the elliptic-integral correction
`T(θ_max) = T₀ · (2/π) · K(sin²(θ_max/2))` — necessary because the
recordings are at finite amplitude (40–90°) but the inertia formula
`I = m·g·d·T²/(4π²)` is the small-amplitude one.

`derive_pendulum_friction(fit)` reads the pendulum geometry (mass, COM,
I_com) from `pendulum_geometry` (which parses `urdf/model.urdf`) and
combines it with the fit to produce viscous and Coulomb friction. It
also reports both `inertia_predicted_kg_m2` (CAD: m·d² + I_com) and
`inertia_measured_kg_m2` (from T₀: m·g·d/ω²), and `validate_free_swing`
warns if they disagree by more than 10% — a sign the URDF is stale or
the recording is contaminated.

Viscous friction uses the CAD-predicted pivot inertia: b = 2·α·I_pred,
where α = b/(2I) is the directly-measured envelope decay. Coulomb
friction comes from the Coulomb-step-per-half-cycle measured in the
envelope fit.

## Recovery and debugging

- **Free-swing fit fails** (too few extrema, no decay, etc.): the recording
  is bad — pendulum didn't move enough, or you released into a vibration
  source. Just retry that run when prompted.
- **Hanging zero drifted between recordings**: the firmware's accumulator
  is monotonic, so this happens if the operator manually wound the
  pendulum past ±180° between runs. Reset the firmware (USB disconnect
  + reconnect) and re-collect.
- **Re-fitting gives different numbers**: expected — small changes in
  the fitter compound. The `freeswing_compare.png` plot is the
  ground-truth: if real and sim envelopes overlap closely, the fit is
  correct regardless of the absolute values.
