# Rig Geometry Parameters

Physical geometry of this rig's two moving bodies (arm, pendulum), extracted
from `urdf/model.urdf` as plain numbers — no 3D model, no mesh files, no
ready-to-load robot description. If you're building your own simulation from
scratch, these are the numbers you need; the model itself (joint tree, mesh
geometry, loader code) is intentionally not provided here.

For per-rig dynamic parameters (motor response, friction) instead of static
geometry, see `sysid_params.json` and `sysid_runbook.md`.

## Convention

Each body's mass/COM/inertia is expressed in that body's own local frame,
with the origin at the joint that attaches it to its parent (standard URDF
convention: `inertial/origin` gives the COM position relative to the link
frame, `inertia` is the inertia tensor about the COM, axis-aligned with the
link frame).

## Bodies

### Arm (motor-driven link)

| Quantity | Value |
|---|---|
| Mass | 0.026 kg |
| COM position (link frame) | x=0.041 m, y=0 m, z=0.011 m |
| Inertia about COM, ixx | 4.29e-6 kg·m² |
| Inertia about COM, iyy | 1.756e-5 kg·m² |
| Inertia about COM, izz | 1.73e-5 kg·m² |
| Inertia about COM, ixz | -1.75e-6 kg·m² |
| Inertia about COM, ixy / iyz | 0 |

### Pendulum (free-swinging link)

| Quantity | Value |
|---|---|
| Mass | 0.014 kg |
| COM position (link frame) | x=0.062 m, y=0 m, z=-0.051 m |
| Inertia about COM, ixx | 8.06e-6 kg·m² |
| Inertia about COM, iyy | 8.55e-6 kg·m² |
| Inertia about COM, izz | 6.1e-7 kg·m² |
| Inertia about COM, iyz | 1.9e-7 kg·m² |
| Inertia about COM, ixy / ixz | 0 |

Note: the pendulum's swing axis is x. The x-component of its COM position is
therefore along the rotation axis and does not affect swing dynamics — only
the y/z components matter for the pendulum's effective moment arm.

## Joints

### base_to_arm (motor-actuated)

| Quantity | Value |
|---|---|
| Type | revolute, actuated |
| Origin offset from base | x=0, y=0, z=0.075 m |
| Origin rotation from base | rpy = 0, 0, -1.57079 rad |
| Rotation axis | z |

### arm_to_pendulum (free-swinging)

| Quantity | Value |
|---|---|
| Type | continuous, unactuated |
| Origin offset from arm | x=0, y=0, z=0.014 m |
| Origin rotation from arm | rpy = 0, 0, 0 |
| Rotation axis | x |

## Independent cross-check values

These come from physical measurements on this rig, independent of the CAD
numbers above — useful to sanity-check your own model once built:

- Measured pendulum free-swing period: ≈ 490 ms (10 half-swings in 2.45 s) →
  natural frequency ω_n ≈ 12.8 rad/s.
- Measured pendulum balance point: ≈ 51 mm from the pivot.
- Cross-check between measured period and CAD inertia: `m·g·d/ω²` gives
  I_axis ≈ 4.48e-5 kg·m², matching the CAD-derived `m·d² + I_com` (≈ 4.46e-5
  kg·m²) to within 1%.

## Not included here

- `urdf/model.urdf` itself (joint tree + mesh references, ready to load into
  a simulator).
- Mesh files (`meshes/*.dae`, `*.stl`) — visual/collision geometry, not
  needed for a physically-correct dynamics model.
- `pendulum_geometry.py` — the parser that extracts these numbers from the
  URDF at runtime.
