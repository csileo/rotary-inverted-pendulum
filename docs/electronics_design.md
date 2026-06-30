# Electronics Design Notes

Why each component on the [BOM](BOM.md) was chosen, and what to know
when re-sourcing or substituting it. The BOM is the procurement
reference; this doc is the *why* behind it.

## Wiring diagram

<img src="../diagrams/system-without-batteries.jpg" height="600">

All components live on a single 40 × 60 mm protoboard. The diagram
above is the canonical layout; component-level photos are in
[`../diagrams/`](../diagrams/).

## Microcontroller — Arduino Nano (ATmega328P, 16 MHz)

- 32 KB flash / 2 KB SRAM is enough for the [LowLevelServer binary
  protocol](../RotaryInvertedPendulum-arduino/LowLevelServer/LowLevelServer.ino)
  plus AS5600 I/O plus AccelStepper at 1 kHz internal step rate. The
  on-device PID variant fits too, with margin.
- Heavy lifting (RL inference, MPC, sysid) runs on the host PC. The
  Nano shuttles state and commands at 2 Mbaud — it doesn't need MCU
  horsepower.
- USB-C variant preferred over Mini/Micro for connector durability —
  the rig gets re-plugged frequently during firmware iteration.
- Any CH340-based clone works; driver is built into modern macOS / Linux.

## Stepper motor — NEMA17 17HS4023 (1 A rated, 22 mm body)

- **Under-loaded by design.** The arm + pendulum is <50 g and the only
  rotational inertia the motor fights is the arm itself (~1.5 × 10⁻⁵
  kg·m²). Phase current rarely exceeds ~0.3 A. The 1 A rating is
  ~3× headroom against ever stalling.
- **Short-body 17HS4023 over the longer 17HS4401.** The motor is
  bolted vertically, so its mass is below the rotation axis and
  doesn't load the bearings — but the shorter body still trims rig
  height and cost. With the loads this rig sees, the longer motor's
  extra torque is wasted.
- Substituting a heavier or stronger motor would add mass and cost
  without benefit; substituting a smaller one (e.g. NEMA14) risks
  losing steps under sudden swing-up commands.

## Stepper driver — DRV8825

- **Vref set to 0.45 V → ~0.9 A current limit** per phase (90 % of
  the motor's 1 A rating). Standard 10 % margin keeps the driver and
  motor below thermal limits indefinitely.
- 8.2–45 V supply range; 12 V chosen as the lowest sensible voltage —
  see "Power supply" below.
- **A4988** is a drop-in alternative but tops out at lower current and
  is audibly louder.
- **TMC2209** would be quieter via internal interpolation but adds UART
  configuration complexity for negligible benefit on this rig: the 8×
  microstepping output from AccelStepper is already smooth at the
  speeds we run.
- **Set Vref before installing the motor.** With the driver powered
  and the motor disconnected, probe Vref against GND while turning
  the trim pot.

### Vref for alternative drivers

If you swap to A4988 or TMC2209, the Vref-trim procedure is the same
but the relation between Vref and the resulting phase-current limit
differs:

| Driver  | Imax → Vref                                                                                                           | Vref @ 0.9 A target                                   |
| ------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| DRV8825 | `Vref = Imax / 2` (Rcs = 0.1 Ω, standard on Pololu and most clones)                                                   | **0.45 V** (we run 0.45 V — close enough)            |
| A4988   | `Vref = Imax × 8 × Rcs`. Pololu carriers use Rcs = 0.05 Ω; some clones use 0.1 Ω — check yours                        | **0.36 V** (Pololu) / **0.72 V** (Rcs = 0.1 Ω clones) |
| TMC2209 | RMS-current calc is non-trivial — use the [TMC220X Vref calculator](https://printpractical.github.io/VrefCalculator/) | per calculator                                        |

## Power supply — 12 V, 2 A

The ideal supply for this rig. Reasoning:

The DRV8825 chops coil current, so supply current ≠ motor phase
current. Power balance:

```
I_supply ≈ (I_phase × V_coil) / V_supply
        ≈ (0.9 A × ~3.5 V) / 12 V ≈ 0.26 A per phase
```

Both phases active plus ~50 mA of logic (Nano + AS5600 + indicators)
gives **~0.6 A steady-state**, with brief peaks to ~1 A on direction
reversals. 2 A is ~3× headroom — the right margin for a cheap
wall-wart with no wasted capacity.

3 A and 5 A adapters work fine (verified empirically) but the extra
current is unused. What actually matters more than headline amps:

- **Regulation quality** — a clean 2 A unit beats a noisy 5 A one for
  ripple. The Nano's 5 V LDO and the AS5600's I²C bus get unhappier
  with messy rails than with low-rated ones.
- **Bulk decoupling on the board** — the 22 µF on the rail handles
  the worst of the chopping spikes. If you ever see brown-outs on
  direction reversals, add a 470 µF near the driver before upsizing
  the adapter.
- **Connector contact** — a loose 5.5 mm barrel jack drops volts under
  spike load regardless of adapter rating.

**Why 12 V** specifically:
- DRV8825 accepts 8.2–45 V; 12 V is the cheapest sensible choice.
- The Nano's onboard linear regulator dissipates 12 V → 5 V
  comfortably. 24 V starts to cook it (the regulator runs hot enough
  to derate above ~16 V continuous).
- 12 V wall-warts with 5.5 mm barrel plugs are ubiquitous.

## Magnetic encoder — AS5600

- **12-bit absolute angle** → 2π / 4096 rad ≈ 0.088° resolution.
  Quantisation is modelled in [`pendulum_env.py`](../RotaryInvertedPendulum-python/src/rl/pendulum_env.py)
  (`PENDULUM_LSB_RAD`) so the policy sees the same step size sim
  and real.
- **Contactless / magnetic** → zero friction on the pendulum joint,
  which is the mechanical DOF we most care about preserving. A
  contact pot or quadrature wheel would add a friction term we'd
  have to identify and randomize against.
- I²C at 400 kHz reads in <1 ms — fits comfortably in the control budget.
- The TZT-style AliExpress modules ship with a small diametrically-
  magnetised disc; no separate magnet sourcing needed.
- **Magnet alignment matters.** Disc face 0.5–3 mm from chip face,
  axially aligned. The AS5600's `AGC` (automatic gain control)
  register reports magnet strength — check it on first power-up;
  out-of-range readings indicate a misaligned or wrong-grade magnet.

## Decoupling — 100 nF ceramic + 22 µF electrolytic

Two-stage decoupling on the 12 V rail at the driver's VMOT pin:

- **100 nF ceramic (104)** handles high-frequency spikes from the
  driver's ~30 kHz chopping. The ceramic's low ESR matters more than
  its capacity at this point.
- **22 µF electrolytic** handles bulk current draw between switching
  cycles. 22 µF is fine for this rig's modest loads; if you upsize
  the motor or see brown-outs on fast reversals, bump to 470 µF
  before upsizing the supply.

## Hookup wire — 26 AWG solid-core

- Single gauge across signals + power, because:
  - Rig peak current is ~1 A; 26 AWG handles 2.2 A continuously in
    chassis wiring.
  - One stock is easier to manage than separate gauges for signal vs
    power, and the difference doesn't matter at these currents.
  - Solid-core terminates more reliably in protoboard plated holes
    than stranded.
- A dedicated thinner gauge for I²C would be overkill at the AS5600
  cable's ~100 mm length.

## Power switch + barrel jack

- An **inline SPST rocker** on the 12 V rail is far more convenient
  than yanking the barrel plug. Power-cycling is a frequent diagnostic
  during firmware development.
- **Jack/plug size mismatch** (5.5 × 2.1 mm jack vs 5.5 × 2.5 mm
  adapter plug): the 0.4 mm pin-diameter difference produces a slightly
  loose fit but reliable contact in practice. If you can find a
  matched 2.5 mm jack at the same price, prefer it — otherwise the
  mismatch is harmless.

## Things deliberately *not* on the BOM

- **Battery / boost converter.** This rig is computer-tethered for the
  RL pipeline; portability isn't a goal. The on-device PID firmware
  *could* be battery-powered, but cheap LiPo + buck is more
  diagnostics surface than the use case warrants.
- **TVS / protection diodes on the rail.** The wall-wart adapters used
  here are well-behaved; an RC snubber on the motor leads or a TVS at
  VMOT would be belt-and-braces but isn't load-bearing for stable
  operation.
- **Logic-level shifters.** Nano (5 V) + AS5600 (3.3 V tolerant on I²C
  with internal pull-ups to 5 V works on this module) — no shifter
  needed. Other AS5600 boards may differ; check the breakout's pull-up
  voltage before assuming.

## Related

- [`BOM.md`](BOM.md) — procurement reference (suppliers, prices, qty).
- [`3d_printing.md`](3d_printing.md) — printing settings and the
  coin-pause technique for the pendulum link.
- [`sysid_runbook.md`](sysid_runbook.md) — measurement protocol that
  validates the electronics chain works end-to-end.
