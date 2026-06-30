# Bill of Materials

Components needed to build one rotary inverted pendulum. Last updated 2026-05-25.

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

## Electronics

| Item                   | Spec                                            | Price (£) | Source                                                              | Notes                                                                                                                                                                  |
| ---------------------- | ----------------------------------------------- | --------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Microcontroller        | Arduino Nano (ATmega328P, 16 MHz, CH340, USB-C) | £1.60     | [AliExpress](https://www.aliexpress.com/item/1005006053215107.html) | Any clone works; original 32 KB flash / 2 KB SRAM constraints assumed in firmware.                                                                                     |
| Stepper motor          | NEMA17 17HS4023, 1 A rated, 22 mm body          | £4.48     | [AliExpress](https://www.aliexpress.com/item/1005006111249881.html) | Short-body variant.                                                                                                                                                    |
| Stepper driver         | DRV8825 (or A4988 / TMC2209)                    | £1.43     | [AliExpress](https://www.aliexpress.com/item/10000278156894.html)   | Set Vref to 0.45 V (≈ 0.9 A current limit, 90 % of motor rating).                                                                                                     |
| Magnetic encoder       | AS5600 12-bit on I²C (with diametric magnet)    | £1.04     | [AliExpress](https://www.aliexpress.com/item/1005006349632569.html) | Reads pendulum angle. RobTillaart Arduino library. Module ships with a small diametrically-magnetised disc that sits on the pendulum shaft end facing the AS5600 face. |
| Power adapter          | 12 V, 2 A barrel (5.5 × 2.1 mm, UK plug)        | £2.59     | [AliExpress](https://www.aliexpress.com/item/1005006467110035.html) | Powers the 12 V rail; Arduino 5 V comes from the Nano's regulator.                                                                                                     |
| DC barrel jack         | 5.5 × 2.1 mm panel-mount socket                 | £0.12     | [AliExpress](https://www.aliexpress.com/item/1005003324016159.html) | Board-side power input.                                                                                                                                                |
| Power switch           | SPST rocker, 20 mm round, 12 V DC               | £1.24     | [AliExpress](https://www.aliexpress.com/item/1005005944839290.html) | Inline on the 12 V rail; turns the rig on/off without unplugging.                                                                                                      |
| 100 nF ceramic cap     | 104 monolithic ceramic                          | £0.01     | [AliExpress](https://www.aliexpress.com/item/1005005691676032.html) | Across the 12 V rail near the driver.                                                                                                                                  |
| 22 µF electrolytic cap | 25 V, aluminium electrolytic                    | £0.02     | [AliExpress](https://www.aliexpress.com/item/1005005945738204.html) | Bulk decoupling on the 12 V rail.                                                                                                                                      |
| Protoboard             | 40 × 60 mm, double-sided                        | £0.26     | [AliExpress](https://www.aliexpress.com/item/1005005945712659.html) | All electronics live here.                                                                                                                                             |
| Header pins            | Female, 2.54 mm pitch, 1×40 strip               | £1.12     | [AliExpress](https://www.aliexpress.com/item/1005006034877497.html) | For mounting the Nano + driver as removable modules.                                                                                                                   |

## Mechanical

| Item                  | Spec                                                             | Price (£)         | Source                                                              | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------- | ---------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ball bearing          | **Bones Reds 608** (8 × 22 × 7 mm precision skate bearing)       | £2.50             | Amazon UK                                                           | Single bearing at the arm/pendulum joint, after simplifying away the dual-bearing design. Skate bearings are designed for low friction and trivially deshield-and-relube.                                                                                                                                                                                                                                                                                  |
| Ball bearing (budget) | 608ZZ 8 × 22 × 7 mm (shielded — **avoid the 608RS variant**)     | £0.18             | [AliExpress](https://www.aliexpress.com/item/1005005778152535.html) | Workable budget alternative. The same listing also sells a 608RS (rubber-sealed) variant — the RS bearings had noticeably more drag/stiction and should be avoided. ABEC-7 marking on bulk lots is meaningless; precision skate bearings (Bones Reds or equivalent) are still preferred for lowest friction.                                                                                                                                               |
| 3D-printed parts      | Arm, base, lid, pendulum link (with embedded coin), encoder boss | ~£2.00 (filament) | —                                                                   | Self-printed. <100 g PLA at ~£20/kg. STL files in `meshes/`; [OnShape source](https://cad.onshape.com/documents/fa8afe5031ca70c78442e408/w/5519455d45464bacd4cf9b1d/e/79273ac76c3305af463951de). The whole assembly is **press-fit** — no fasteners required. Pendulum link mass, COM, and inertia tensor live in `urdf/model.urdf` (exported from Onshape with PLA-density-corrected materials) — the RL sim and the sysid pipeline both read from there. |
| 2p coin (UK)          | UK 2-pence piece, 7.12 g, 25.91 × 2.03 mm                        | £0.02             | —                                                                   | Press-fit into the pendulum link as the tip mass. Its mass, COM, and inertia are baked into the pendulum link inertia exported from Onshape — see `urdf/model.urdf`.                                                                                                                                                                                                                                                                                       |
| Motor heatsink        | Aluminium extruded, NEMA17 / 42 mm pattern                       | £0.55             | [AliExpress](https://www.aliexpress.com/item/4000723868050.html)    | Sticks onto the back face of the motor. Keeps the motor cool during extended training sessions.                                                                                                                                                                                                                                                                                                                                                            |

## Tools (reference, not part of unit cost)

- Soldering iron + 60/40 leaded solder (or lead-free if preferred)
- 3D printer (PLA / PETG, 0.4 mm nozzle, 0.2 mm layer)
- Multimeter (continuity + Vref pot trim)
- [30 AWG solid-core wire kit](https://www.amazon.co.uk/gp/product/B0C2Z4FNN5) (5-roll set, Amazon UK)
