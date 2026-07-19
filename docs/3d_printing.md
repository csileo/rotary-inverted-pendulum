# 3D Printing Notes

Practical notes for printing the mechanical parts. STL files live in
[`../meshes/`](../meshes); the OnShape source is linked from
[`BOM.md`](BOM.md).

## Settings

PLA or PETG, 0.4 mm nozzle, 0.2 mm layer (Bambu Studio defaults). Use
'Tree' supports for the enclosure (base, lid) and the arm.

## Pendulum link — pause printing to insert the coin

The pendulum link has a slot sized for a UK 2-pence coin (7.12 g) as the
tip mass. Configure your slicer to **pause on layer 21** — drop the coin
into the slot, then resume. The next layers seal it in.

If you skip the pause, the link prints with an empty void where the coin
should be: its mass and inertia then disagree with what
[`urdf/model.urdf`](../urdf/model.urdf) says, and the policy will not
transfer from sim.

## Soldering note

Cut your hookup wires slightly longer than you think you'll need. Excess
length is trivial to manage; a wire that's a few millimetres too short
turns the next solder joint into a fight.
