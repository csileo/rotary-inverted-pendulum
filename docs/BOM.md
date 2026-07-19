# Bill of Materials

Components needed to build one rotary inverted pendulum. Last updated 2026-05-25.

## Sommaire

This document has **two alternative sourcing sections** for the same build —
pick one, don't mix them casually (specs/tolerances differ between listings):

- [AliExpress / UK (original)](#aliexpress--uk-original) — the original
  upstream sourcing, cheapest.
- [Amazon France (as-built sourcing)](#amazon-france-as-built-sourcing) —
  what was actually used to build this specific rig, fast delivery from
  France.

## AliExpress / UK (original)

**Cost estimate**  
One complete rig comes in at **under £20** in parts (~£14 electronics + ~£5 mechanical) — over 230× cheaper than the £4,500 Quanser QUBE.

**Design rationale**  
For *why* each electronics component was chosen (power supply
sizing, motor selection, driver tradeoffs, decoupling, etc.), see
[`electronics_design.md`](electronics_design.md). This BOM is a pure
procurement reference.

**Suppliers**  
Unless otherwise noted, items were sourced from AliExpress.
The only exception is the Bones Reds skate bearing (Amazon UK).

**Variant selection**  
Several AliExpress listings sell multiple variants from a single
product page (e.g., bearing ZZ vs RS, stepper body length, switch
colour/rating, with/without included accessories). The page's default
selection is **not always** the right one — always check the **Spec**
column below and pick the matching variant before adding to cart.

### Electronics

| Item                   | Spec                                            | Price (£) | Source                                                              | Notes                                                                                                                                                                  |
| ---------------------- | ----------------------------------------------- | --------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Microcontroller        | Arduino Nano (ATmega328P, 16 MHz, CH340, USB-C) | £1.60     | [AliExpress](https://www.aliexpress.com/item/1005006053215107.html) | Any clone works; original 32 KB flash / 2 KB SRAM constraints assumed in firmware.                                                                                     |
| Stepper motor          | NEMA17 17HS4023, 1 A rated, 22 mm body          | £4.48     | [AliExpress](https://www.aliexpress.com/item/1005006111249881.html) | Short-body variant.                                                                                                                                                    |
| Stepper driver         | DRV8825 (or A4988 / TMC2209)                    | £1.43     | [AliExpress](https://www.aliexpress.com/item/10000278156894.html)   | Set Vref to 0.485 V (≈ 0.9 A current limit, 90 % of motor rating).                                                                                                     |
| Magnetic encoder       | AS5600 12-bit on I²C (with diametric magnet)    | £1.04     | [AliExpress](https://www.aliexpress.com/item/1005006349632569.html) | Reads pendulum angle. RobTillaart Arduino library. Module ships with a small diametrically-magnetised disc that sits on the pendulum shaft end facing the AS5600 face. |
| Power adapter          | 12 V, 2 A barrel (5.5 × 2.1 mm, UK plug)        | £2.59     | [AliExpress](https://www.aliexpress.com/item/1005006467110035.html) | Powers the 12 V rail; Arduino 5 V comes from the Nano's regulator.                                                                                                     |
| DC barrel jack         | 5.5 × 2.1 mm panel-mount socket                 | £0.12     | [AliExpress](https://www.aliexpress.com/item/1005003324016159.html) | Board-side power input.                                                                                                                                                |
| Power switch           | SPST rocker, 20 mm round, 12 V DC               | £1.24     | [AliExpress](https://www.aliexpress.com/item/1005005944839290.html) | Inline on the 12 V rail; turns the rig on/off without unplugging.                                                                                                      |
| 100 nF ceramic cap     | 104 monolithic ceramic                          | £0.01     | [AliExpress](https://www.aliexpress.com/item/1005005691676032.html) | Across the 12 V rail near the driver.                                                                                                                                  |
| 22 µF electrolytic cap | 25 V, aluminium electrolytic                    | £0.02     | [AliExpress](https://www.aliexpress.com/item/1005005945738204.html) | Bulk decoupling on the 12 V rail.                                                                                                                                      |
| Protoboard             | 40 × 60 mm, double-sided                        | £0.26     | [AliExpress](https://www.aliexpress.com/item/1005005945712659.html) | All electronics live here.                                                                                                                                             |
| Header pins            | Female, 2.54 mm pitch, 1×40 strip               | £1.12     | [AliExpress](https://www.aliexpress.com/item/1005006034877497.html) | For mounting the Nano + driver as removable modules.                                                                                                                   |

### Mechanical

| Item                  | Spec                                                             | Price (£)         | Source                                                              | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------- | ---------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ball bearing          | **Bones Reds 608** (8 × 22 × 7 mm precision skate bearing)       | £2.50             | Amazon UK                                                           | Single bearing at the arm/pendulum joint, after simplifying away the dual-bearing design. Skate bearings are designed for low friction and trivially deshield-and-relube.                                                                                                                                                                                                                                                                                  |
| Ball bearing (budget) | 608ZZ 8 × 22 × 7 mm (shielded — **avoid the 608RS variant**)     | £0.18             | [AliExpress](https://www.aliexpress.com/item/1005005778152535.html) | Workable budget alternative. The same listing also sells a 608RS (rubber-sealed) variant — the RS bearings had noticeably more drag/stiction and should be avoided. ABEC-7 marking on bulk lots is meaningless; precision skate bearings (Bones Reds or equivalent) are still preferred for lowest friction.                                                                                                                                               |
| 3D-printed parts      | Arm, base, lid, pendulum link (with embedded coin), encoder boss | ~£2.00 (filament) | —                                                                   | Self-printed. <100 g PLA at ~£20/kg. STL files in `meshes/`; [OnShape source](https://cad.onshape.com/documents/fa8afe5031ca70c78442e408/w/5519455d45464bacd4cf9b1d/e/79273ac76c3305af463951de). The whole assembly is **press-fit** — no fasteners required. Pendulum link mass, COM, and inertia tensor live in `urdf/model.urdf` (exported from Onshape with PLA-density-corrected materials) — the RL sim and the sysid pipeline both read from there. |
| 2p coin (UK)          | UK 2-pence piece, 7.12 g, 25.91 × 2.03 mm                        | £0.02             | —                                                                   | Press-fit into the pendulum link as the tip mass. Its mass, COM, and inertia are baked into the pendulum link inertia exported from Onshape — see `urdf/model.urdf`.                                                                                                                                                                                                                                                                                       |
| Motor heatsink        | Aluminium extruded, NEMA17 / 42 mm pattern                       | £0.55             | [AliExpress](https://www.aliexpress.com/item/4000723868050.html)    | Sticks onto the back face of the motor. Keeps the motor cool during extended training sessions.                                                                                                                                                                                                                                                                                                                                                            |

### Tools (reference, not part of unit cost)

- Soldering iron + 60/40 leaded solder (or lead-free if preferred)
- 3D printer (PLA / PETG, 0.4 mm nozzle, 0.2 mm layer)
- Multimeter (continuity + Vref pot trim)
- [30 AWG solid-core wire kit](https://www.amazon.co.uk/gp/product/B0C2Z4FNN5) (5-roll set, Amazon UK)

---

## Amazon France (as-built sourcing)

The sections above are the original upstream BOM and are left as-is. This
section documents how **this specific rig** was actually sourced — mostly
Amazon.fr instead of AliExpress/Amazon UK, plus a few parts added along the
way (post-incident protection, in-arm sensor wiring, a stripboard swap).

**Cost estimate** — Electronics ~€220 + mechanical ~€32 for this basket.
This is bulk-kit list pricing (caps, resistors, headers, crimp terminals,
wire all bought as multi-use kits), not a true marginal per-rig cost — most
kits cover several builds.

### Electronics

| Item                    | Spec                                                       | Price (€) | Source                                                | Notes                                                                                                                                                                     |
| ----------------------- | ------------------------------------------------------------ | --------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Microcontroller         | Arduino Nano (ATmega328P, 16 MHz, CH340, USB-C)               | €11.99    | [Amazon.fr](https://www.amazon.fr/dp/B0D8WGSPW3)        | AZDelivery, pre-soldered + USB-C cable.                                                                                                                                 |
| Stepper motor           | NEMA17 17HE08-1004S, 1 A rated, 23 mm body                    | €15.26    | [Amazon.fr](https://www.amazon.fr/dp/B0B93PNYCP)        | STEPPERONLINE pancake, 17 Ncm, 1.8°/step. 23 mm vs. 22 mm spec — no impact (checked against STL).                                                                        |
| Stepper driver          | DRV8825 + heatsink (pack of 2)                                | €6.99     | [Amazon.fr](https://www.amazon.fr/dp/B08BF9KTN9)        | AZDelivery. Vref ≈ 0.485 V. 1 used, 1 spare.                                                                                                                             |
| Magnetic encoder        | AS5600 12-bit on I²C, pack of 2, with diametric magnet         | €9.49     | [Amazon.fr](https://www.amazon.fr/dp/B0B778L7T3)        | Hailege clone — behaves differently from the original Seeed module (direct TWI polling + 100 kHz bus needed instead of the Wire lib; see firmware). 1 unit damaged 09/07 (frozen I²C bus), 1 spare to install. |
| Power adapter           | 12 V, 2.5 A barrel (5.5 × 2.1 mm, EU plug)                     | €16.00    | [Amazon.fr](https://www.amazon.fr/dp/B09LGZ3SVP)        | —                                                                                                                                                                        |
| DC barrel jack           | 5.5 × 2.1 mm panel-mount socket                                | €8.99     | [Amazon.fr](https://www.amazon.fr/dp/B07ZTJXHC6)        | Board-side power input. Wired via crimp terminals, see below.                                                                                                             |
| Power switch             | SPST rocker, 20 mm round, 12 V DC (pack of 10)                 | €8.99     | [Amazon.fr](https://www.amazon.fr/dp/B09VZ74QCL)        | Wired via crimp terminals, see below.                                                                                                                                     |
| Crimp terminal kit       | 280-pc assorted spade/ring/fork terminals, M+F                 | €9.98     | [Amazon.fr](https://www.amazon.fr/dp/B09WDT6469)        | Added vs. original BOM. Quick-disconnect crimp wiring to the power switch and barrel jack instead of direct soldering — keeps that wiring serviceable.                    |
| 100 nF ceramic cap       | 104 monolithic ceramic (kit of 360, 12 values)                 | €6.99     | [Amazon.fr](https://www.amazon.fr/dp/B0CLRLBW19)        | 1 used from kit.                                                                                                                                                          |
| 22 µF electrolytic cap   | 25 V, aluminium electrolytic (kit of 630, 24 values)            | €17.99    | [Amazon.fr](https://www.amazon.fr/dp/B07PN5P64W)        | 1 used from kit. Superseded near the driver by the 220 µF/35 V cap below.                                                                                                 |
| 220 µF electrolytic cap  | 35 V, aluminium electrolytic (kit)                              | €7.99     | [Amazon.fr](https://www.amazon.fr/dp/B0CMQCG3M8)        | Added vs. original BOM. Added after the 05/07 USB incident — bulk decoupling at the DRV8825; the 22 µF/25 V cap above only reaches 25 V, not enough headroom.          |
| Series resistors         | 1/4 W kit, 25 values (1 Ω – 1 MΩ)                               | €8.99     | [Amazon.fr](https://www.amazon.fr/dp/B0BTP88DZN)        | Added vs. original BOM. 3× 100–220 Ω in series on STEP/DIR/EN — post-incident protection.                                                                              |
| TVS diode                | 1.5KE18A, unidirectional                                       | €4.60     | [Amazon.fr](https://www.amazon.fr/dp/B0DD4F8L2W)        | Added vs. original BOM. On the 12 V rail near the DRV8825 — post-incident protection.                                                                                  |
| USB isolator             | 5 kV galvanic isolation (ISOUSB211-based)                       | —         | [Amazon.fr](https://www.amazon.fr/dp/B0FR4SW7ML)        | Added vs. original BOM. Jhoinrch RH-07B, between PC and Arduino — the one part the PC's safety actually depends on after the incident.                                 |
| Protoboard               | 50 × 100 mm, single-sided copper stripboard (pack of 25)        | €17.42    | [Amazon.fr](https://www.amazon.fr/dp/B085WJ7535)        | **Replaces the original double-sided 40×60 mm protoboard.** Rademacher stripboard, 2.54 mm grid — matches the actual soldered layout (see `diagrams/`).                  |
| Header pins               | Male + female, 2.54 mm pitch, 40-pin single row (pack of 30)   | €11.71    | [Amazon.fr](https://www.amazon.fr/dp/B07DBY753C)        | For mounting the Nano and DRV8825 as removable/socketed modules.                                                                                                          |
| Wire (main harness)       | 22 AWG solid-core, 6 colours × 9 m                              | €19.99    | [Amazon.fr](https://www.amazon.fr/dp/B07V5FVSYL)        | —                                                                                                                                                                          |
| Wire (in-arm sensor)      | 30 AWG solid-core, 6 colours × 30 m                             | €19.99    | [Amazon.fr](https://www.amazon.fr/dp/B0C2Z4FNN5)        | Promoted from the "Tools" list above to a proper component row. Used for the AS5600 wiring **inside the arm** — thinner/more flexible than the 22 AWG harness, needed to route through the arm without adding stiffness at the joint. |

### Mechanical

| Item              | Spec                                              | Price (€) | Source                                            | Notes                                                                                                          |
| ------------------ | ---------------------------------------------------| --------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Ball bearing        | 608ZZ 8 × 22 × 7 mm, shielded (pack of 20)          | €9.45     | [Amazon.fr](https://www.amazon.fr/dp/B09TQG47KQ)    | QWORK. Budget option (not Bones Reds).                                                                          |
| Bearing grease      | White lithium grease spray, 400 ml                  | €12.95    | [Amazon.fr](https://www.amazon.fr/dp/B01G12I1VU)    | Added vs. original BOM. WD-40 Specialist — relube for the 608ZZ bearings after deshielding. One can covers many rebuilds. |
| Motor heatsink      | Aluminium, 40 × 40 × 11 mm (pack of 4)               | €8.79     | [Amazon.fr](https://www.amazon.fr/dp/B0BFW4T8SW)    | —                                                                                                                |
| 3D-printed parts    | unchanged from original                              | ~€2 (filament) | — | Self-printed. Unchanged.                                                                                       |
| 2p coin (UK)        | unchanged from original                              | €0.50 × 2 | eBay.fr                                             | Unchanged.                                                                                                       |

### Tools (reference, not part of unit cost)

- Soldering iron + solder — Miniware TS101 or Yihua station
- 3D printer: Bambu Lab A1 mini + PLA Basic filament
- Multimeter (continuity + Vref pot trim)
- Post-incident protection parts (USB isolator, TVS diode, series resistors, 220 µF cap) are listed under Electronics above, not here — they're part of every build now, not optional tooling.
