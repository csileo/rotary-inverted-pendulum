"""Rebuild system-without-batteries-reinforced-protection.drawio from stripboard_matrix.csv.

system-without-batteries.drawio already has good-looking art for the Arduino
Nano, DRV8825, AS5600, motor, switch and DC socket (hand-built vector shapes,
not photos, apart from two small externally-fetched textures). But its board
layer ("Protoboard - 4x6 cm") has the wrong physical size/aspect for the real
19x15-hole stripboard, the DRV8825 is positioned off the correct grid relative
to the Nano, and the internal wiring is drawn as stylized bent cable art
instead of the straight strip-to-strip jumps a real stripboard has.

This script keeps the Nano/AS5600/motor/switch/DC-socket art untouched,
translates the DRV8825 group by the offset needed to land its real pins back
on the correct grid (computed from stripboard_matrix.csv, verified against
the DIR/STP/EN/VMOT anchors already in the source file), regenerates the
board layer as an accurate 19x15 hole grid with copper strips and cuts, and
regenerates the wiring layer from scratch: every net in the CSV (fN wires,
the 5V/GND/VMOT rails, R1/R2/R3 with their resistor bodies, the C1-2 cap pair,
the TVS diode, and the external AS5600/motor/power stubs) as a straight line
directly between grid coordinates, plus a simple USB-isolator box spliced
into the Nano's USB cable.

Coordinate convention (matches the existing Nano/DRV8825 art already in the
source file - a CSV "row" is a pin position along a header, drawn along the
picture's X axis; a CSV "column" is which header/rail, drawn along Y):
    X(row)  = X0 + X_SCALE * row_index(row)
    Y(col)  = Y0 + Y_SCALE * col_number(col)
with 30 drawio units per 2.54mm hole, matching the pin pitch already baked
into the reused Nano/DRV8825 art.

Usage:
    python tools/diagrams/gen_stripboard_realistic_diagram.py [--csv PATH] [--src-drawio PATH] [--out PATH]
"""

import argparse
import base64
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_stripboard_diagram import (  # noqa: E402
    BOARD_ROWS, COMPONENTS, EXTERNAL, EXTERNAL_COLOR, OVERHANG_ROWS,
    REAL_PIN_COLUMNS, WIRE_COLOR, load_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGRAMS = REPO_ROOT / "diagrams"
DEFAULT_CSV = DIAGRAMS / "system-without-batteries-reinforced-protection-matrix.csv"
DEFAULT_SRC_DRAWIO = DIAGRAMS / "system-without-batteries.drawio"
DEFAULT_OUT = DIAGRAMS / "system-without-batteries-reinforced-protection.drawio"

# Real photos for the two capacitors and the TVS diode, in place of
# schematic-style shapes. The 220uF electrolytic and the TVS diode are stock
# photos extracted from a hand-arranged version of this same .drawio (saved
# locally since their own source is otherwise unavailable); the 100nF ceramic
# cap is kept as a live external URL as hand-picked in that same arrangement.
CAP_220UF_IMG = DIAGRAMS / "cap_220uf.png"
TVS_IMG = DIAGRAMS / "tvs_diode.png"
CAP_104_URL = (
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.xuanxcapacitors.com"
    "%2Fwp-content%2Fuploads%2F2023%2F06%2F104-ceramic-capacitor.png&amp;f=1&amp;nofb=1"
    "&amp;ipt=756745bcdc6e0407c91e0825e00a72c6aa6189f6ee8977f28cfc64efee81d9ec&amp;ipo=images"
)


def image_data_uri(path):
    mime = {".png": "png", ".webp": "webp", ".jpg": "jpeg", ".jpeg": "jpeg"}[Path(path).suffix.lower()]
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime},{b64}"

# Grid scale: 30 drawio units per 2.54mm hole, matching the pin pitch already
# used by the reused Nano/DRV8825 art (confirmed from their real wire/pin
# coordinates in the source file: e.g. D2/D9/DIR/STP/EN/VMOT are all exactly
# 30 units apart per pin).
PITCH = 30
X0 = 4880   # X(row_index=0), i.e. row "R-3"
Y0 = 745    # Y(col_number=0), i.e. "Col00" (Nano top header)

# DRV8825's DIR pin (row R07, Col10) as currently placed in the source file -
# used to compute how far the DRV8825 group must be translated to land back
# on the correct grid. (Independently verified against STP/EN/VMOT: all four
# need the exact same translation, confirming the DRV8825 body art itself is
# internally consistent and just needs a rigid shift.)
DRV_DIR_CURRENT = (4460.0, 1075.0)
DRV_GROUP_ID = "YWhdqIxVLoHPOZUmhBRt-1018"
NANO_GROUP_ID = "YWhdqIxVLoHPOZUmhBRt-755"

PROTOBOARD_LAYER_ID = "Pd-qLGOYEHb491aJz7A3-33"
PROTOBOARD_LAYER_MARKER = f'<mxCell id="{PROTOBOARD_LAYER_ID}"'
DRV8825_LAYER_MARKER = '<mxCell id="YWhdqIxVLoHPOZUmhBRt-841"'
NANO_LAYER_MARKER = '<mxCell id="YWhdqIxVLoHPOZUmhBRt-1" value="Arduino Nano"'
DC_SOCKET_LAYER_MARKER = '<mxCell id="jaUYTfBtadA-Ni3pfydp-4"'
WIRING_LAYER_MARKER = '<mxCell id="1" value="wiring"'
NOTES_LAYER_MARKER = '<mxCell id="bxmNM9AHO8O3NAPSHGuk-1"'

# A dedicated "pins" layer, added last (on top of everything, including the
# board strips) so the Nano/DRV8825 pin names stay crisply legible in solid
# black - independent of the opacity=55 ghosting applied to the Nano/DRV
# bodies themselves and of the strips (Protoboard layer) drawing over them.
# The original in-body pin-label cells (light grey, part of the Nano/DRV art)
# are removed so they don't linger as invisible duplicates underneath.
PINS_LAYER_ID = "2"
NANO_PIN_LABEL_IDS_OLD = [
    "YWhdqIxVLoHPOZUmhBRt-792", "YWhdqIxVLoHPOZUmhBRt-793", "YWhdqIxVLoHPOZUmhBRt-794",
    "YWhdqIxVLoHPOZUmhBRt-795", "YWhdqIxVLoHPOZUmhBRt-796", "YWhdqIxVLoHPOZUmhBRt-797",
    "YWhdqIxVLoHPOZUmhBRt-798", "YWhdqIxVLoHPOZUmhBRt-799", "YWhdqIxVLoHPOZUmhBRt-800",
    "YWhdqIxVLoHPOZUmhBRt-801", "YWhdqIxVLoHPOZUmhBRt-802", "YWhdqIxVLoHPOZUmhBRt-803",
    "YWhdqIxVLoHPOZUmhBRt-804", "YWhdqIxVLoHPOZUmhBRt-805", "YWhdqIxVLoHPOZUmhBRt-806",
    "YWhdqIxVLoHPOZUmhBRt-807", "YWhdqIxVLoHPOZUmhBRt-808", "YWhdqIxVLoHPOZUmhBRt-809",
    "YWhdqIxVLoHPOZUmhBRt-810", "YWhdqIxVLoHPOZUmhBRt-811", "YWhdqIxVLoHPOZUmhBRt-812",
    "YWhdqIxVLoHPOZUmhBRt-813", "YWhdqIxVLoHPOZUmhBRt-814", "YWhdqIxVLoHPOZUmhBRt-815",
    "YWhdqIxVLoHPOZUmhBRt-816", "YWhdqIxVLoHPOZUmhBRt-817", "YWhdqIxVLoHPOZUmhBRt-818",
    "YWhdqIxVLoHPOZUmhBRt-819", "YWhdqIxVLoHPOZUmhBRt-820", "YWhdqIxVLoHPOZUmhBRt-821",
]
DRV_PIN_LABEL_IDS_OLD = [
    "YWhdqIxVLoHPOZUmhBRt-957", "YWhdqIxVLoHPOZUmhBRt-958", "YWhdqIxVLoHPOZUmhBRt-959",
    "YWhdqIxVLoHPOZUmhBRt-960", "YWhdqIxVLoHPOZUmhBRt-993", "YWhdqIxVLoHPOZUmhBRt-994",
    "YWhdqIxVLoHPOZUmhBRt-995", "YWhdqIxVLoHPOZUmhBRt-996", "YWhdqIxVLoHPOZUmhBRt-997",
    "YWhdqIxVLoHPOZUmhBRt-998", "YWhdqIxVLoHPOZUmhBRt-999", "YWhdqIxVLoHPOZUmhBRt-1000",
    "YWhdqIxVLoHPOZUmhBRt-1001", "YWhdqIxVLoHPOZUmhBRt-1002", "YWhdqIxVLoHPOZUmhBRt-1003",
    "YWhdqIxVLoHPOZUmhBRt-1004",
    # the two blank chip-pin decoration marks (within the nested DRV8825 chip
    # body group) that were promoted to DRV_PIN_BLANKS_OLD in the pins layer
    "YWhdqIxVLoHPOZUmhBRt-975", "YWhdqIxVLoHPOZUmhBRt-989",
]
# (value, absolute x, absolute y) - Nano never moves, so these are fixed.
NANO_PIN_LABELS = [
    ("A0", 4780, 755), ("A1", 4750, 755), ("A2", 4720, 755), ("A3", 4690, 755),
    ("A5", 4630, 755), ("A6", 4600, 755), ("A7", 4570, 755), ("A4", 4660, 755),
    ("RST", 4510, 755), ("GND", 4480, 755), ("5V", 4540, 755), ("VIN", 4450, 755),
    ("REF", 4810, 755), ("3V3", 4840, 755), ("D13", 4870, 755),
    ("D9", 4780, 895), ("D8", 4750, 895), ("D7", 4720, 895), ("D6", 4690, 895),
    ("D4", 4630, 895), ("D3", 4600, 895), ("D2", 4570, 895), ("D5", 4660, 895),
    ("RST", 4510, 895), ("RX0", 4480, 895), ("GND", 4540, 895), ("TX1", 4450, 895),
    ("D10", 4810, 895), ("D11", 4840, 895), ("D12", 4870, 895),
]
# (value, x, y) in the *original* (untranslated) frame - DRV8825 moves by the
# same (dx, dy) computed for DRV_GROUP_ID, so these get that delta applied.
DRV_PIN_LABELS_OLD = [
    ("M1", 4600, 1085), ("M0", 4630, 1085), ("EN", 4660, 1085), ("M2", 4570, 1085),
    ("SLP", 4510, 1085), ("STP", 4480, 1085), ("DIR", 4450, 1085), ("RST", 4540, 1085),
    ("GND", 4450, 1195), ("FLT", 4480, 1195), ("1A", 4540, 1195), ("1B", 4570, 1195),
    ("2A", 4510, 1195), ("GND", 4630, 1195), ("VMOT", 4660, 1195), ("2B", 4600, 1195),
]
# Two small blank (invisible) markers that came along with the DRV pin labels
# in the hand-arranged file - no fill/stroke/text, kept for exact fidelity.
DRV_PIN_BLANKS_OLD = [(4551, 1184), (4485, 1184)]

# The Switch image and the Motor group both live directly under the "wiring"
# layer in the source file (parent="1"), so wiping that layer to rebuild the
# wiring from scratch also deletes them. Both are re-spliced back in,
# unmodified (beyond the repositioning below), at the position computed here.
SWITCH_ID = "jaUYTfBtadA-Ni3pfydp-1"
MOTOR_GROUP_ID = "zqDiZcR5qHbJr3BYcA1d-16"
AS5600_GROUP_ID = "n6NpK2DIO6W0D-b5wM51-78"
DC_SOCKET_GROUP_ID = "jaUYTfBtadA-Ni3pfydp-46"

# The "AS5600 magnetic encoder" and "DC female socket" captions are separate
# text cells living directly under their component's layer (siblings of the
# group, not its children), so translating the group alone leaves them
# behind - each needs its own translate_group() call with the same delta.
AS5600_LABEL_ID = "HLAI94WT9K0ZCO-hjb5A-158"
DC_SOCKET_LABEL_ID = "jaUYTfBtadA-Ni3pfydp-51"

# Connection points on components whose art is reused as-is, as originally
# placed in system-without-batteries.drawio (extracted by hand from their
# group's local pin-label/shape coordinates + the group's own absolute
# geometry - see conversation notes). AS5600 pin dots sit just off its chip
# body; DC-socket terminals are its two metal prongs; the motor's 4-wire
# connector is a single tight cluster near the bottom of its photo.
AS5600_GROUP_OLD = (4960.0, 400.0, 270.0, 270.0)
AS5600_PINS_OLD = {
    "AVCC": (5015.0, 505.0),   # VCC
    "AGND": (5015.0, 565.0),   # GND
    "ASCL": (5175.0, 520.0),   # SCL
    "ASDA": (5175.0, 550.0),   # SDA
}
SWITCH_BOX_OLD = (4900.0, 1165.0, 120.0, 120.0)  # x, y, w, h
DC_SOCKET_GROUP_OLD = (5020.0, 970.0, 200.0, 140.0)
DC_SOCKET_LABEL_OLD = (5000.0, 1110.0)
DC_SOCKET_POS_TERMINAL_OLD = (5040.0, 1070.0)  # -44, wired to the switch
DC_SOCKET_GND_TERMINAL_OLD = (5020.0, 1013.0)  # -45, wired straight to TGND
MOTOR_GROUP_OLD = (4850.0, 1410.0, 340.0, 380.0)

# Final hand-arranged positions (dragged into place in the drawio desktop app
# so AS5600/motor/switch/socket wires stay clear of the board and each other -
# these are absolute targets, not derived from a gap/centering formula, since
# that's how they were actually placed). main() translates each component from
# its *_OLD position to its *_TARGET by a plain rigid delta.
AS5600_GROUP_TARGET = (4340.0, 390.0)
MOTOR_GROUP_TARGET = (4870.0, 1370.0)
SWITCH_TARGET = (4370.0, 1620.0)  # x deliberately matches the T12V board anchor's x,
                                  # so the switch-to-board wire is a plain vertical line
DC_SOCKET_GROUP_TARGET = (4470.0, 1380.0)
DC_SOCKET_LABEL_TARGET = (4510.0, 1530.0)

# The motor's 4-wire connector is a tight cluster of leads on its photo, not
# evenly spaced - these are per-wire offsets from the motor group's top-left
# corner, hand-picked to land each wire on its actual lead. M1/M2 were
# originally attached (in the hand-arranged file) to the motor photo itself
# via entryX/Y fractions - and that photo has rotation=90, so the fraction is
# defined in its *unrotated* local frame; these offsets are the fraction
# rotated back into absolute board coordinates (verified against M3/M4, which
# are plain unattached points and land right next to these on the connector).
MOTOR_CONNECTOR_OFFSETS = {
    "M1": (50.0, 150.0),
    "M2": (50.0, 160.0),
    "M3": (51.0, 173.0),
    "M4": (51.0, 186.0),
}

# M1/M2/switch/DC-socket-gnd are live-attached (source=/target=) to their
# component in the hand-arranged file, so their *other* endpoint is a stale
# cached fallback point (from wherever that component sat mid-edit) that
# render-time attachment resolution overrides and is otherwise never used.
# These are reproduced byte-for-byte purely for file fidelity, not because
# the numbers mean anything on their own.
MOTOR_IMAGE_ID = "K1kWtO_6Q8ll-fQjAVFB-48"
DC_SOCKET_GND_TERMINAL_ID = "jaUYTfBtadA-Ni3pfydp-45"
STALE_CACHE = {
    "M1_target": (4850.0, 1520.0),
    "M2_target": (4850.0, 1540.0),
    "switch_to_board_source": (4370.0, 1610.0),
    "switch_to_socket_source": (4470.0, 1637.0),
    "dc_socket_gnd_target": (4125.0, 1298.0),
}

_id_counter = [0]


def next_id(prefix):
    _id_counter[0] += 1
    return f"{prefix}-{_id_counter[0]}"


def row_index(label, row_labels):
    return row_labels.index(label)


def X(row_label, row_labels):
    return X0 - PITCH * row_index(row_label, row_labels)


def col_number(col_name):
    return int(col_name[3:])


def Y(col_name):
    return Y0 + PITCH * col_number(col_name)


def pt(row_label, col_name, row_labels):
    return X(row_label, row_labels), Y(col_name)


def board_edges(row_labels, col_names):
    x_right = X(BOARD_ROWS[0], row_labels) + PITCH / 2
    x_left = X(BOARD_ROWS[-1], row_labels) - PITCH / 2
    y_top = Y(col_names[0]) - PITCH / 2
    y_bottom = Y(col_names[-1]) + PITCH / 2
    return x_left, x_right, y_top, y_bottom


ID_PARENT_RE = re.compile(r'<mxCell id="([^"]+)"[^>]*?parent="([^"]*)"')


def build_children_map(text):
    children = {}
    for m in ID_PARENT_RE.finditer(text):
        cid, parent = m.group(1), m.group(2)
        children.setdefault(parent, []).append(cid)
    return children


def descendants(children_map, root_id):
    result = []
    stack = list(children_map.get(root_id, []))
    while stack:
        cur = stack.pop()
        result.append(cur)
        stack.extend(children_map.get(cur, []))
    return result


def extract_block(text, cell_id):
    m = re.search(r'<mxCell id="' + re.escape(cell_id) + r'"', text)
    if not m:
        raise ValueError(f"cell {cell_id} not found")
    start = m.start()
    gt_idx = text.index(">", start)
    if text[gt_idx - 1] == "/":
        return text[start:gt_idx + 1] + "\n"
    end = text.index("</mxCell>", gt_idx) + len("</mxCell>")
    return text[start:end] + "\n"


def extract_with_descendants(text, children_map, root_id):
    ids = [root_id] + descendants(children_map, root_id)
    return "".join(extract_block(text, i) for i in ids)


def remove_cell(text, cell_id):
    """Delete a single cell's block (its whole line(s)) from the document -
    used to strip the original in-body Nano/DRV8825 pin-label cells once
    they've been replaced by fresh copies in the dedicated "pins" layer."""
    m = re.search(r'<mxCell id="' + re.escape(cell_id) + r'"', text)
    if not m:
        raise ValueError(f"cell {cell_id} not found")
    start = m.start()
    gt_idx = text.index(">", start)
    end = gt_idx + 1 if text[gt_idx - 1] == "/" else text.index("</mxCell>", gt_idx) + len("</mxCell>")
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.index("\n", end) + 1
    return text[:line_start] + text[line_end:]


def pin_label_cell(value, x, y):
    cid = next_id("rppin")
    return (
        f'        <mxCell id="{cid}" value="{value}" style="text;html=1;align=center;verticalAlign=middle;'
        f'whiteSpace=wrap;rounded=0;rotation=180;fontColor=#000000;fontFamily=Tahoma;fontStyle=1;fontSize=10;" '
        f'vertex="1" parent="{PINS_LAYER_ID}">\n'
        f'          <mxGeometry x="{_fmt(x)}" y="{_fmt(y)}" width="20" height="20" as="geometry" />\n'
        f'        </mxCell>\n'
    )


def pin_blank_cell(x, y):
    cid = next_id("rppinblank")
    return (
        f'        <mxCell id="{cid}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;'
        f'strokeColor=none;rotation=90;align=center;verticalAlign=middle;fontFamily=Tahoma;fontSize=10;'
        f'fontColor=#000000;fontStyle=1;gradientColor=none;" vertex="1" parent="{PINS_LAYER_ID}">\n'
        f'          <mxGeometry x="{_fmt(x)}" y="{_fmt(y)}" width="3.130434782608696" height="14.117647058823529" as="geometry" />\n'
        f'        </mxCell>\n'
    )


def translate_group(text, group_id, dx, dy):
    pattern = re.compile(
        r'(<mxCell id="' + re.escape(group_id) + r'"[^>]*>\s*<mxGeometry x=")([\-\d.]+)(" y=")([\-\d.]+)(")'
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"group {group_id} not found")
    new_x = float(m.group(2)) + dx
    new_y = float(m.group(4)) + dy
    return text[: m.start()] + m.group(1) + _fmt(new_x) + m.group(3) + _fmt(new_y) + m.group(5) + text[m.end():]


def add_style_property(text, cell_id, prop):
    """Insert a style property (e.g. "opacity=55;") into a cell's style="..."
    attribute, so the whole group (and everything nested in it) is rendered
    at reduced opacity - used to make the Nano/DRV8825 art see-through so the
    board grid underneath stays visible."""
    pattern = re.compile(r'(<mxCell id="' + re.escape(cell_id) + r'"[^>]*style=")')
    m = pattern.search(text)
    if not m:
        raise ValueError(f"cell {cell_id} not found")
    return text[: m.end()] + prop + text[m.end():]


def replace_layer_subtree(text, layer_marker, next_layer_marker, new_children_xml):
    layer_start = text.index(layer_marker)
    layer_line_end = text.index("\n", layer_start) + 1
    next_start = text.index(next_layer_marker)
    # back up to the start of that cell's line
    next_line_start = text.rfind("\n", 0, next_start) + 1
    return text[:layer_line_end] + new_children_xml + text[next_line_start:]


def reorder_layers(text, order_markers, region_end_marker):
    """Reorders whole layer blocks (each layer's own declaration cell plus
    all its children, currently sitting back-to-back) into a new order.
    order_markers gives the desired order and must list every layer marker
    currently present in the contiguous span from the first of them (in
    document order) up to region_end_marker (the marker right after the
    span, e.g. the next layer's declaration)."""
    positions = sorted((text.index(m), m) for m in order_markers)
    region_start = text.rfind("\n", 0, positions[0][0]) + 1
    region_end = text.rfind("\n", 0, text.index(region_end_marker)) + 1
    blocks = {}
    for i, (pos, marker) in enumerate(positions):
        block_start = text.rfind("\n", 0, pos) + 1
        block_end = text.rfind("\n", 0, positions[i + 1][0]) + 1 if i + 1 < len(positions) else region_end
        blocks[marker] = text[block_start:block_end]
    new_region = "".join(blocks[m] for m in order_markers)
    return text[:region_start] + new_region + text[region_end:]


def _fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def wire_edge(x1, y1, x2, y2, color, width=3, parent="1"):
    cid = next_id("rpw")
    return (
        f'        <mxCell id="{cid}" value="" style="endArrow=none;html=1;rounded=0;'
        f'strokeWidth={width};strokeColor={color};" parent="{parent}" edge="1">\n'
        f'          <mxGeometry relative="1" as="geometry">\n'
        f'            <mxPoint x="{_fmt(x1)}" y="{_fmt(y1)}" as="sourcePoint" />\n'
        f'            <mxPoint x="{_fmt(x2)}" y="{_fmt(y2)}" as="targetPoint" />\n'
        f'          </mxGeometry>\n'
        f'        </mxCell>\n'
    )


def wire_edge_attached(x1, y1, x2, y2, color, source=None, target=None,
                        fraction_prefix="entry", fx=None, fy=None, extra_style="",
                        width=3, parent="1"):
    """Like wire_edge, but with a source=/target= shape attachment plus its
    exit/entry fraction - used for the handful of wires that are attached to
    a specific component (not a plain floating point) in the hand-arranged
    file. x1,y1,x2,y2 are still written as the cached fallback mxPoints
    (superseded at render time by the live attachment on whichever end has
    source=/target=), so a full round-trip through the drawio editor matches
    byte-for-byte."""
    cid = next_id("rpw")
    frac_style = ""
    if fx is not None:
        fx_s = f"{fx:.3f}".rstrip("0").rstrip(".")
        fy_s = f"{fy:.3f}".rstrip("0").rstrip(".")
        frac_style = f'{fraction_prefix}X={fx_s};{fraction_prefix}Y={fy_s};{fraction_prefix}Dx=0;{fraction_prefix}Dy=0;'
    attach_attr = ""
    if source:
        attach_attr = f' source="{source}"'
    elif target:
        attach_attr = f' target="{target}"'
    return (
        f'        <mxCell id="{cid}" value="" style="endArrow=none;html=1;rounded=0;'
        f'strokeWidth={width};strokeColor={color};{frac_style}{extra_style}" parent="{parent}" edge="1"{attach_attr}>\n'
        f'          <mxGeometry relative="1" as="geometry">\n'
        f'            <mxPoint x="{_fmt(x1)}" y="{_fmt(y1)}" as="sourcePoint" />\n'
        f'            <mxPoint x="{_fmt(x2)}" y="{_fmt(y2)}" as="targetPoint" />\n'
        f'          </mxGeometry>\n'
        f'        </mxCell>\n'
    )


def label(x, y, text, color="#000000", size=11, align="center"):
    cid = next_id("rpl")
    return (
        f'        <mxCell id="{cid}" value="{text}" style="text;html=1;align={align};'
        f'verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor={color};fontSize={size};" '
        f'parent="1" vertex="1">\n'
        f'          <mxGeometry x="{_fmt(x - 40)}" y="{_fmt(y - 10)}" width="80" height="20" as="geometry" />\n'
        f'        </mxCell>\n'
    )


def resistor_body(xm, ym, x1, y1, x2, y2, text, color="#d9c9a3"):
    """A small rotated rectangle centered on the wire midpoint, standing in for
    a real axial resistor - oriented along the wire direction."""
    cid = next_id("rpres")
    dx, dy = x2 - x1, y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    w, h = 26, 11
    return (
        f'        <mxCell id="{cid}" value="{text}" style="rounded=1;whiteSpace=wrap;html=1;'
        f'fillColor={color};strokeColor=#8a7350;strokeWidth=1;fontSize=8;rotation={_fmt(angle)};" '
        f'parent="1" vertex="1">\n'
        f'          <mxGeometry x="{_fmt(xm - w / 2)}" y="{_fmt(ym - h / 2)}" width="{w}" height="{h}" as="geometry" />\n'
        f'        </mxCell>\n'
    )


def image_shape(x, y, w, h, image_src, rotation=0, flip_h=None, clip_path=None):
    """A shape=image cell, styled to match what the drawio desktop app writes
    when an image is pasted in directly (verticalLabelPosition/
    labelBackgroundColor/verticalAlign, rotation omitted when 0). image_src is
    either a data URI (image_data_uri()) or a plain external URL (CAP_104_URL
    is kept live rather than downloaded, matching how it was originally
    hand-picked). clip_path is a CSS inset(...) string, used to crop a source
    photo down to just the component (drawio applies this at render time, no
    local re-cropping needed). flip_h: None omits flipH, True/False writes
    flipH=1/0 explicitly (matches whether that cell's flip was ever toggled
    in the editor, even back off)."""
    extra = ""
    if rotation:
        extra += f"rotation={_fmt(rotation)};"
    if clip_path:
        extra += f"clipPath={clip_path};"
    if flip_h is not None:
        extra += f"flipH={1 if flip_h else 0};"
    cid = next_id("rpimg")
    return (
        f'        <mxCell id="{cid}" value="" style="shape=image;verticalLabelPosition=bottom;'
        f'labelBackgroundColor=default;verticalAlign=top;aspect=fixed;imageAspect=0;'
        f'image={image_src};{extra}" parent="1" vertex="1">\n'
        f'          <mxGeometry x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" as="geometry" />\n'
        f'        </mxCell>\n'
    )


def build_board_xml(row_labels, col_names, grid):
    out = []
    ncols = len(col_names)

    # copper strips: one per physical board row, a thin vertical bar at X(row),
    # spanning the uncut Y(col) run(s) - a stripboard's copper trace runs along
    # the column axis at a fixed row.
    for r in BOARD_ROWS:
        x = X(r, row_labels)
        run_start = None
        for c in col_names:
            cell = grid.get((r, c), "-")
            if cell == "//":
                if run_start is not None:
                    out.append(_strip_rect(x, Y(run_start) - PITCH * 0.35, Y(_prev_col(col_names, c)) + PITCH * 0.35))
                run_start = None
            else:
                if run_start is None:
                    run_start = c
        if run_start is not None:
            out.append(_strip_rect(x, Y(run_start) - PITCH * 0.35, Y(col_names[-1]) + PITCH * 0.35))

    # holes
    for r in row_labels:
        x = X(r, row_labels)
        for c in col_names:
            cell = grid.get((r, c), "-")
            if r in OVERHANG_ROWS and cell == "-":
                continue
            cid = next_id("rphole")
            out.append(
                f'        <mxCell id="{cid}" value="" style="ellipse;whiteSpace=wrap;html=1;'
                f'fillColor=#FFFFFF;strokeColor=#888888;strokeWidth=1;" parent="{PROTOBOARD_LAYER_ID}" vertex="1">\n'
                f'          <mxGeometry x="{_fmt(x - 4)}" y="{_fmt(Y(c) - 4)}" width="8" height="8" as="geometry" />\n'
                f'        </mxCell>\n'
            )
            if cell == "X":
                d = 5
                out.append(wire_edge(x - d, Y(c) - d, x + d, Y(c) + d, "#c0392b", width=2, parent=PROTOBOARD_LAYER_ID))
                out.append(wire_edge(x - d, Y(c) + d, x + d, Y(c) - d, "#c0392b", width=2, parent=PROTOBOARD_LAYER_ID))

    # cut marks
    for r in BOARD_ROWS:
        x = X(r, row_labels)
        for c in col_names:
            if grid.get((r, c)) == "//":
                cid = next_id("rpcut")
                out.append(
                    f'        <mxCell id="{cid}" value="" style="endArrow=none;html=1;rounded=0;'
                    f'strokeWidth=2;strokeColor=#ff0000;" parent="{PROTOBOARD_LAYER_ID}" edge="1">\n'
                    f'          <mxGeometry relative="1" as="geometry">\n'
                    f'            <mxPoint x="{_fmt(x - 5)}" y="{_fmt(Y(c))}" as="sourcePoint" />\n'
                    f'            <mxPoint x="{_fmt(x + 5)}" y="{_fmt(Y(c))}" as="targetPoint" />\n'
                    f'          </mxGeometry>\n'
                    f'        </mxCell>\n'
                )

    # board outline
    x_left, x_right, y_top, y_bottom = board_edges(row_labels, col_names)
    cid = next_id("rpboard")
    out.append(
        f'        <mxCell id="{cid}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;'
        f'strokeColor=#1a4fa0;strokeWidth=3;" parent="{PROTOBOARD_LAYER_ID}" vertex="1">\n'
        f'          <mxGeometry x="{_fmt(x_left)}" y="{_fmt(y_top)}" width="{_fmt(x_right - x_left)}" '
        f'height="{_fmt(y_bottom - y_top)}" as="geometry" />\n'
        f'        </mxCell>\n'
    )
    return "".join(out)


def _prev_col(col_names, c):
    i = col_names.index(c)
    return col_names[i - 1]


def _strip_rect(x, y1, y2):
    cid = next_id("rpstrip")
    return (
        f'        <mxCell id="{cid}" value="" style="rounded=0;whiteSpace=wrap;html=1;'
        f'fillColor=#ead9c2;strokeColor=none;" parent="{PROTOBOARD_LAYER_ID}" vertex="1">\n'
        f'          <mxGeometry x="{_fmt(x - 3)}" y="{_fmt(y1)}" width="6" height="{_fmt(y2 - y1)}" as="geometry" />\n'
        f'        </mxCell>\n'
    )


def find_pair(grid, code, row_labels, col_names, exclude_cols=frozenset()):
    hits = []
    for r in row_labels:
        for c in col_names:
            if c in exclude_cols:
                continue
            if grid.get((r, c)) == code:
                hits.append((r, c))
    return hits


def compute_external_anchors(row_labels, col_names, grid):
    anchors = {}
    for code in sorted(EXTERNAL):
        hits = find_pair(grid, code, row_labels, col_names, exclude_cols=REAL_PIN_COLUMNS)
        if len(hits) != 1:
            continue
        (r, c) = hits[0]
        anchors[code] = pt(r, c, row_labels)
    return anchors


def build_wiring_xml(row_labels, col_names, grid, layout):
    out = []

    # Collect R1-R3 and the fN wires as plain segments first, so overlapping
    # ones (sharing a column -> same Y, or R1 sharing a row -> same X) can be
    # nudged apart like stripboard-layout-proposal.jpg does, before any of
    # them are actually drawn.
    segments = []
    for code in sorted(COMPONENTS - {"C1-2", "TVS"}):
        hits = find_pair(grid, code, row_labels, col_names)
        if len(hits) != 2:
            continue
        (r1, c1), (r2, c2) = hits
        x1, y1 = pt(r1, c1, row_labels)
        x2, y2 = pt(r2, c2, row_labels)
        segments.append({"code": code, "kind": "resistor", "x1": x1, "y1": y1,
                          "x2": x2, "y2": y2, "color": "#000000"})

    wire_codes = sorted({v for v in grid.values() if v.startswith("f") and v[1:].isdigit()})
    for code in wire_codes:
        hits = find_pair(grid, code, row_labels, col_names)
        if len(hits) != 2:
            continue
        (r1, c1), (r2, c2) = hits
        x1, y1 = pt(r1, c1, row_labels)
        x2, y2 = pt(r2, c2, row_labels)
        segments.append({"code": code, "kind": "wire", "x1": x1, "y1": y1,
                          "x2": x2, "y2": y2, "color": WIRE_COLOR.get(code, "#2266cc")})

    OFFSET_STEP = 6
    horiz, vert = {}, {}
    for seg in segments:
        if seg["y1"] == seg["y2"]:
            horiz.setdefault(round(seg["y1"]), []).append(seg)
        elif seg["x1"] == seg["x2"]:
            vert.setdefault(round(seg["x1"]), []).append(seg)
    for groups, keys in ((horiz, ("y1", "y2")), (vert, ("x1", "x2"))):
        for segs in groups.values():
            k = len(segs)
            if k <= 1:
                continue
            for i, seg in enumerate(sorted(segs, key=lambda s: s["code"])):
                off = (i - (k - 1) / 2) * OFFSET_STEP
                seg[keys[0]] += off
                seg[keys[1]] += off

    for seg in segments:
        x1, y1, x2, y2, color = seg["x1"], seg["y1"], seg["x2"], seg["y2"], seg["color"]
        if seg["kind"] == "resistor":
            out.append(wire_edge(x1, y1, x2, y2, "#000000", width=2))
            out.append(resistor_body((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2, "220 Ω"))
        else:
            out.append(wire_edge(x1, y1, x2, y2, color))

    # C1-2 capacitor pair - real photos (220uF electrolytic + 100nF ceramic),
    # no separate connecting line: the two photos sit directly over the net's
    # two holes and read as the physical component bridging them.
    hits = find_pair(grid, "C1-2", row_labels, col_names)
    if len(hits) == 2:
        (r1, c1), (r2, c2) = hits
        x1, y1 = pt(r1, c1, row_labels)
        x2, y2 = pt(r2, c2, row_labels)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        out.append(image_shape(xm - 30, ym - 55, 60, 60, image_data_uri(CAP_220UF_IMG), flip_h=True))
        out.append(image_shape(xm - 15, ym - 15, 30, 30, CAP_104_URL,
                                clip_path="inset(10% 25.52% 41.67% 27.27%)"))

    # TVS diode - real photo (a diagonal stock photo, not an axial crop), the
    # part number isn't legible on it so a small synthetic text label is
    # overlaid on top. rotation=136 is hand-tuned to this specific photo's
    # inherent lead angle so its leads end up horizontal, matching this net's
    # two anchors (same column, a row apart -> a horizontal connection).
    hits = find_pair(grid, "TVS", row_labels, col_names)
    if len(hits) == 2:
        (r1, c1), (r2, c2) = hits
        x1, y1 = pt(r1, c1, row_labels)
        x2, y2 = pt(r2, c2, row_labels)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        out.append(image_shape(xm - 17, ym - 16.22, 33, 32.43, image_data_uri(TVS_IMG),
                                rotation=136, flip_h=False, clip_path="inset(29% 30.47% 33.33% 24.61%)"))
        cid = next_id("rplbl")
        out.append(
            f'        <mxCell id="{cid}" value="&lt;font style=&quot;font-size: 4px; color: rgb(230, 230, 230);&quot;&gt;'
            f'P6KE18A&lt;/font&gt;" style="text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;'
            f'rounded=0;rotation=0;fontColor=#FFFFFF;fontFamily=Tahoma;fontStyle=1;fontSize=10;" '
            f'parent="1" vertex="1">\n'
            f'          <mxGeometry x="{_fmt(xm - 26)}" y="{_fmt(ym - 13.22)}" width="50" height="20" as="geometry" />\n'
            f'        </mxCell>\n'
        )

    # external wires - each routed all the way to the real component it's
    # meant to reach, using the connection points established from the
    # reused (untouched, but repositioned - see main()) AS5600/switch/
    # DC-socket/motor art.
    anchors = compute_external_anchors(row_labels, col_names, grid)

    # AS5600: AVCC/AGND/ASCL/ASDA straight to their pin dot on the chip
    for code, dest in layout["as5600_pins"].items():
        if code not in anchors:
            continue
        x, y = anchors[code]
        out.append(wire_edge(x, y, dest[0], dest[1], EXTERNAL_COLOR.get(code, "#000000")))

    # Motor: M1-M4 straight to the motor's 4-wire connector. M1/M2 are
    # live-attached to the motor photo itself (target=MOTOR_IMAGE_ID) in the
    # hand-arranged file, so their drawn coordinate is only a stale cached
    # fallback, reproduced as-is for fidelity; M3/M4 are plain points.
    motor_pins = layout["motor_connector_pins"]
    motor_fractions = {"M1": (0.441, 0.853), "M2": (0.471, 0.853)}
    for code, (dest_x, dest_y) in motor_pins.items():
        if code not in anchors:
            continue
        x, y = anchors[code]
        color = EXTERNAL_COLOR.get(code, "#000000")
        if code in motor_fractions:
            fx, fy = motor_fractions[code]
            sx, sy = STALE_CACHE[f"{code}_target"]
            out.append(wire_edge_attached(x, y, sx, sy, color, target=MOTOR_IMAGE_ID,
                                           fx=fx, fy=fy, extra_style="entryPerimeter=0;"))
        else:
            out.append(wire_edge(x, y, dest_x, dest_y, color))

    # Power: T12V board anchor -> switch -> DC socket (+) terminal;
    # TGND board anchor -> DC socket (-) terminal directly - matching the
    # switch being inline with the positive rail only, as in system-without-batteries.jpg.
    # The switch's x is deliberately aligned with the T12V anchor's x (see
    # SWITCH_TARGET), so a plain vertical line reaches the board without
    # cutting across the DC socket. Both switch legs are live-attached
    # (source=SWITCH_ID) in the hand-arranged file, and the GND leg is
    # live-attached to the socket's ground terminal (target=
    # DC_SOCKET_GND_TERMINAL_ID) - each has one stale cached endpoint,
    # reproduced as-is for fidelity.
    if "T12V" in anchors:
        x, y = anchors["T12V"]
        switch_to_socket = layout["switch_to_socket"]
        dc_pos = layout["dc_socket_pos"]
        sbx, sby = STALE_CACHE["switch_to_board_source"]
        ssx, ssy = STALE_CACHE["switch_to_socket_source"]
        out.append(wire_edge_attached(sbx, sby, x, y, EXTERNAL_COLOR["T12V"],
                                       source=SWITCH_ID, fraction_prefix="exit", fx=0, fy=0.5))
        out.append(wire_edge_attached(ssx, ssy, dc_pos[0], dc_pos[1], EXTERNAL_COLOR["T12V"],
                                       source=SWITCH_ID, fraction_prefix="exit", fx=1, fy=0.5))
    if "TGND" in anchors:
        x, y = anchors["TGND"]
        gtx, gty = STALE_CACHE["dc_socket_gnd_target"]
        out.append(wire_edge_attached(x, y, gtx, gty, EXTERNAL_COLOR["TGND"],
                                       target=DC_SOCKET_GND_TERMINAL_ID, fx=1, fy=0.75))

    # USB isolator spliced into the Nano's USB cable, next to the actual USB
    # port (just right of the D10-D13 overhang pins, between the two header
    # rows in height).
    usb_port_x = X(row_labels[0], row_labels) + 60
    usb_port_y = (Y("Col00") + Y("Col06")) / 2
    usb_x = usb_port_x + 150
    usb_y = usb_port_y
    cid = next_id("rpusb")
    out.append(wire_edge(usb_port_x, usb_port_y, usb_x - 40, usb_y, "#999999", width=4))
    out.append(
        f'        <mxCell id="{cid}" value="USB ISOLATOR&#xa;5 kV" style="rounded=1;whiteSpace=wrap;html=1;'
        f'fillColor=#222222;strokeColor=#000000;fontColor=#ffffff;fontSize=9;fontStyle=1;" parent="1" vertex="1">\n'
        f'          <mxGeometry x="{_fmt(usb_x - 40)}" y="{_fmt(usb_y - 30)}" width="80" height="60" as="geometry" />\n'
        f'        </mxCell>\n'
    )
    out.append(wire_edge(usb_x + 40, usb_y, usb_x + 160, usb_y, "#999999", width=4))

    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--src-drawio", default=str(DEFAULT_SRC_DRAWIO))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    row_labels, col_names, grid = load_matrix(args.csv)
    text = Path(args.src_drawio).read_text(encoding="utf-8")

    target_dir_x = X("R07", row_labels)
    target_dir_y = Y("Col10")
    dx = target_dir_x - DRV_DIR_CURRENT[0]
    dy = target_dir_y - DRV_DIR_CURRENT[1]
    text = translate_group(text, DRV_GROUP_ID, dx, dy)

    anchors = compute_external_anchors(row_labels, col_names, grid)

    # AS5600 -> hand-arranged position, clear of the board and its wires.
    dx_as5600 = AS5600_GROUP_TARGET[0] - AS5600_GROUP_OLD[0]
    dy_as5600 = AS5600_GROUP_TARGET[1] - AS5600_GROUP_OLD[1]
    text = translate_group(text, AS5600_GROUP_ID, dx_as5600, dy_as5600)
    text = translate_group(text, AS5600_LABEL_ID, dx_as5600, dy_as5600)
    as5600_pins_new = {c: (x + dx_as5600, y + dy_as5600) for c, (x, y) in AS5600_PINS_OLD.items()}

    # Motor -> hand-arranged position under the board.
    dx_motor = MOTOR_GROUP_TARGET[0] - MOTOR_GROUP_OLD[0]
    dy_motor = MOTOR_GROUP_TARGET[1] - MOTOR_GROUP_OLD[1]
    text = translate_group(text, MOTOR_GROUP_ID, dx_motor, dy_motor)
    motor_connector_pins_new = {
        code: (MOTOR_GROUP_TARGET[0] + ox, MOTOR_GROUP_TARGET[1] + oy)
        for code, (ox, oy) in MOTOR_CONNECTOR_OFFSETS.items()
    }

    # Switch and DC socket -> hand-arranged bottom-left position, each moved
    # independently (their offsets from the original file aren't quite the
    # same, so they're not a rigid pair).
    dx_switch = SWITCH_TARGET[0] - SWITCH_BOX_OLD[0]
    dy_switch = SWITCH_TARGET[1] - SWITCH_BOX_OLD[1]
    text = translate_group(text, SWITCH_ID, dx_switch, dy_switch)
    switch_box_new = (SWITCH_TARGET[0], SWITCH_TARGET[1], SWITCH_BOX_OLD[2], SWITCH_BOX_OLD[3])
    switch_to_board_new = (switch_box_new[0], switch_box_new[1] + switch_box_new[3] / 2)
    switch_to_socket_new = (switch_box_new[0] + switch_box_new[2], switch_box_new[1] + switch_box_new[3] / 2)

    dx_socket = DC_SOCKET_GROUP_TARGET[0] - DC_SOCKET_GROUP_OLD[0]
    dy_socket = DC_SOCKET_GROUP_TARGET[1] - DC_SOCKET_GROUP_OLD[1]
    text = translate_group(text, DC_SOCKET_GROUP_ID, dx_socket, dy_socket)
    dx_socket_label = DC_SOCKET_LABEL_TARGET[0] - DC_SOCKET_LABEL_OLD[0]
    dy_socket_label = DC_SOCKET_LABEL_TARGET[1] - DC_SOCKET_LABEL_OLD[1]
    text = translate_group(text, DC_SOCKET_LABEL_ID, dx_socket_label, dy_socket_label)
    dc_socket_pos_new = (DC_SOCKET_POS_TERMINAL_OLD[0] + dx_socket, DC_SOCKET_POS_TERMINAL_OLD[1] + dy_socket)
    dc_socket_gnd_new = (DC_SOCKET_GND_TERMINAL_OLD[0] + dx_socket, DC_SOCKET_GND_TERMINAL_OLD[1] + dy_socket)

    # Ghost the Nano/DRV8825 art (opacity cascades to every nested shape) so
    # the board grid underneath stays visible through the component bodies.
    text = add_style_property(text, NANO_GROUP_ID, "opacity=55;")
    text = add_style_property(text, DRV_GROUP_ID, "opacity=55;")

    # Switch and the Motor group live directly under the "wiring" layer we're
    # about to wipe and rebuild - pull their (now repositioned) XML out first
    # so they can be spliced back in.
    children_map = build_children_map(text)
    preserved_xml = extract_block(text, SWITCH_ID) + extract_with_descendants(text, children_map, MOTOR_GROUP_ID)
    preserved_xml = preserved_xml.replace("stepper motor", "Stepper motor")
    # A zero-length, invisible self-edge on the motor's photo (source=target=
    # MOTOR_IMAGE_ID, no geometry) - a harmless leftover from editing the
    # motor wire attachments in the desktop app, kept for exact fidelity.
    preserved_xml += (
        f'        <mxCell id="{next_id("rpstray")}" style="edgeStyle=none;html=1;" edge="1" '
        f'parent="{MOTOR_GROUP_ID}" source="{MOTOR_IMAGE_ID}" target="{MOTOR_IMAGE_ID}">\n'
        f'          <mxGeometry relative="1" as="geometry" />\n'
        f'        </mxCell>\n'
    )

    board_xml = build_board_xml(row_labels, col_names, grid)
    text = replace_layer_subtree(text, PROTOBOARD_LAYER_MARKER, DRV8825_LAYER_MARKER, board_xml)

    # Layer *document order* controls z-order between layers - the source
    # template has Protoboard/DRV8825/Nano in that order (strips underneath
    # both chips), but the hand-arranged file reverses it to Nano/DRV8825/
    # Protoboard, so the (semi-transparent) chip bodies sit under the strips
    # instead of over them.
    text = reorder_layers(
        text,
        [NANO_LAYER_MARKER, DRV8825_LAYER_MARKER, PROTOBOARD_LAYER_MARKER],
        DC_SOCKET_LAYER_MARKER,
    )

    layout = {
        "as5600_pins": as5600_pins_new,
        "motor_connector_pins": motor_connector_pins_new,
        "switch_to_board": switch_to_board_new,
        "switch_to_socket": switch_to_socket_new,
        "dc_socket_pos": dc_socket_pos_new,
        "dc_socket_gnd": dc_socket_gnd_new,
    }
    # Switch/motor must come first (i.e. render *underneath*) so the motor
    # wires draw on top of the motor's opaque body and stay visible reaching
    # into its connector, instead of being hidden behind it.
    wiring_xml = preserved_xml + build_wiring_xml(row_labels, col_names, grid, layout)
    text = replace_layer_subtree(text, WIRING_LAYER_MARKER, NOTES_LAYER_MARKER, wiring_xml)

    # Strip the original in-body pin-label cells (light grey, faded along with
    # the rest of the Nano/DRV8825 art by opacity=55) and replace them with
    # fresh copies in a new "pins" layer, added last so it renders on top of
    # everything - including the board strips - in solid black.
    for cid in NANO_PIN_LABEL_IDS_OLD + DRV_PIN_LABEL_IDS_OLD:
        text = remove_cell(text, cid)

    pins_xml = [f'        <mxCell id="{PINS_LAYER_ID}" value="pins" style="locked=1;" parent="0"/>\n']
    for value, px, py in NANO_PIN_LABELS:
        pins_xml.append(pin_label_cell(value, px, py))
    for value, px, py in DRV_PIN_LABELS_OLD:
        pins_xml.append(pin_label_cell(value, px + dx, py + dy))
    for px, py in DRV_PIN_BLANKS_OLD:
        pins_xml.append(pin_blank_cell(px + dx, py + dy))
    text = text.replace("      </root>", "".join(pins_xml) + "      </root>")

    # The hand-arranged file has the "notes" layer unlocked (a GUI-state
    # artifact from working in the editor, not a deliberate change).
    text = text.replace(
        '<mxCell id="bxmNM9AHO8O3NAPSHGuk-1" value="notes" style="locked=1;" parent="0" visible="0" />',
        '<mxCell id="bxmNM9AHO8O3NAPSHGuk-1" value="notes" style="" parent="0" visible="0" />',
    )

    Path(args.out).write_text(text, encoding="utf-8")
    print(f"written: {args.out}")
    print(f"AS5600 translated by dx={dx_as5600:.1f} dy={dy_as5600:.1f}")
    print(f"Motor translated by dx={dx_motor:.1f} dy={dy_motor:.1f}")
    print(f"Switch translated by dx={dx_switch:.1f} dy={dy_switch:.1f}")
    print(f"DC socket translated by dx={dx_socket:.1f} dy={dy_socket:.1f}")
    print(f"DRV8825 translated by dx={dx:.1f} dy={dy:.1f}")


if __name__ == "__main__":
    main()
