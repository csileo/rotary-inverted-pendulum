"""Render a stripboard_matrix.csv into a schematic JPG.

Usage:
    python tools/diagrams/gen_stripboard_diagram.py [--csv PATH] [--out PATH] [--dpi N]

Draws simple symbols only (no realistic component shapes): grid holes,
copper strips with cuts, zigzag resistors, basic capacitor/diode marks,
curved wires for intra-board links, and outward stubs for external wires.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "diagrams" / "system-without-batteries-reinforced-protection-matrix.csv"
DEFAULT_OUT = REPO_ROOT / "diagrams" / "stripboard-layout-proposal.jpg"

REAL_PIN_COLUMNS = {"Col00", "Col06", "Col10", "Col15"}
COMPONENTS = {"R1", "R2", "R3", "C1-2", "TVS"}
EXTERNAL = {"ASDA", "ASCL", "AVCC", "AGND", "M1", "M2", "M3", "M4", "T12V", "TGND"}

# Wire colors taken from system-without-batteries-reinforced-protection.jpg:
# orange=5V, black=GND, red=VMOT/12V, magenta=I2C, grey=motor phases.
ORANGE, BLACK, RED, MAGENTA, GREY = "#e08020", "#111111", "#cc0000", "#cc3399", "#666666"
YELLOW, BLUE, GREEN = "#d4b106", "#1a56cc", "#2e8b30"
WIRE_COLOR = {
    "f1": ORANGE, "f2": ORANGE, "f3": ORANGE, "f4": ORANGE, "f5": ORANGE, "f6": ORANGE, "f13": ORANGE,
    "f7": BLACK, "f8": BLACK, "f9": BLACK, "f10": BLACK,
    "f11": RED, "f12": RED,
}
EXTERNAL_COLOR = {
    "ASDA": RED, "ASCL": ORANGE, "AVCC": YELLOW, "AGND": BLACK,
    "M1": RED, "M2": GREEN, "M3": BLACK, "M4": BLUE,
    "T12V": RED, "TGND": BLACK,
}

BOARD_ROWS = [f"R{n:02d}" for n in range(15)]  # R00..R14, physical stripboard
OVERHANG_ROWS = ["R-3", "R-2", "R-1"]  # off-board Nano overhang

NANO_COLS = (0, 6)   # top header col, bottom header col
DRV_COLS = (10, 15)  # top header col, bottom header col


def load_matrix(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_names = header[1:]
        row_labels = []
        grid = {}
        for row in reader:
            if not row or not row[0]:
                continue
            label = row[0]
            row_labels.append(label)
            for col_name, cell in zip(col_names, row[1:]):
                grid[(label, col_name)] = cell
    return row_labels, col_names, grid


def row_y(row_label, row_labels):
    """Top-to-bottom y position: R-3 highest, R15 lowest."""
    idx = row_labels.index(row_label)
    return -idx


def find_pair(grid, code, row_labels, col_names, exclude_cols=frozenset()):
    hits = []
    for r in row_labels:
        for c in col_names:
            if c in exclude_cols:
                continue
            if grid.get((r, c)) == code:
                hits.append((r, c))
    return hits


def draw_zigzag(ax, x1, y1, x2, y2, n=5, amp=0.14, color="black"):
    """Resistor symbol: straight leads on the first/last quarter, zigzag body in the middle."""
    dx, dy = x2 - x1, y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length  # unit vector along the body
    px, py = -uy, ux  # perpendicular unit vector, for zigzag offset

    lead = 0.28
    bx1, by1 = x1 + ux * lead, y1 + uy * lead
    bx2, by2 = x2 - ux * lead, y2 - uy * lead

    ax.plot([x1, bx1], [y1, by1], color=color, linewidth=1.6, zorder=5)
    ax.plot([x2, bx2], [y2, by2], color=color, linewidth=1.6, zorder=5)

    xs, ys = [bx1], [by1]
    for i in range(1, 2 * n):
        t = i / (2 * n)
        bx, by = bx1 + (bx2 - bx1) * t, by1 + (by2 - by1) * t
        off = amp if i % 2 else -amp
        xs.append(bx + px * off)
        ys.append(by + py * off)
    xs.append(bx2)
    ys.append(by2)
    ax.plot(xs, ys, color=color, linewidth=1.6, zorder=5)


def draw_wire(ax, x, y1, y2, offset, color):
    """Continuous straight vertical wire at column x+offset (offset keeps two wires that
    share a column from drawing exactly on top of each other)."""
    x = x + offset
    ax.plot([x, x], [y1, y2], color=color, linewidth=1.5, zorder=4)


def draw_capacitor(ax, x, y1, y2, polarized):
    ymid = (y1 + y2) / 2
    ax.plot([x, x], [y1, ymid - 0.08], color="black", linewidth=1.2, zorder=5)
    ax.plot([x, x], [ymid + 0.08, y2], color="black", linewidth=1.2, zorder=5)
    half = 0.22 if polarized else 0.14
    ax.plot([x - half, x + half], [ymid - 0.08, ymid - 0.08], color="black", linewidth=2.2, zorder=5)
    if polarized:
        ax.plot([x - half, x + half], [ymid + 0.08, ymid + 0.08], color="black", linewidth=1.0, zorder=5)
    else:
        ax.plot([x - half, x + half], [ymid + 0.08, ymid + 0.08], color="black", linewidth=2.2, zorder=5)


def draw_diode(ax, x, y1, y2):
    ymid = (y1 + y2) / 2
    ax.plot([x, x], [y1, ymid - 0.15], color="black", linewidth=1.2, zorder=5)
    ax.plot([x, x], [ymid + 0.15, y2], color="black", linewidth=1.2, zorder=5)
    tri = [(x - 0.16, ymid - 0.15), (x + 0.16, ymid - 0.15), (x, ymid + 0.15)]
    ax.add_patch(plt.Polygon(tri, closed=True, facecolor="white", edgecolor="black", linewidth=1.2, zorder=5))
    ax.plot([x - 0.16, x + 0.16], [ymid + 0.15, ymid + 0.15], color="black", linewidth=2.0, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    row_labels, col_names, grid = load_matrix(args.csv)
    ncols = len(col_names)

    fig, ax = plt.subplots(figsize=(14, 12))

    # --- copper strips (physical board rows only), broken at cuts ---
    for r in BOARD_ROWS:
        y = row_y(r, row_labels)
        run_start = None
        for i, c in enumerate(col_names):
            cell = grid.get((r, c), "-")
            if cell == "//":
                if run_start is not None:
                    ax.add_patch(Rectangle((run_start - 0.3, y - 0.09), (i - run_start) - 0.4, 0.18,
                                            facecolor="#ead9c2", edgecolor="none", zorder=1))
                run_start = i + 1
            else:
                if run_start is None:
                    run_start = i
        if run_start is not None:
            ax.add_patch(Rectangle((run_start - 0.3, y - 0.09), ncols - run_start - 0.4 + 0.3, 0.18,
                                    facecolor="#ead9c2", edgecolor="none", zorder=1))

    # --- all holes ---
    for r in row_labels:
        y = row_y(r, row_labels)
        for i, c in enumerate(col_names):
            cell = grid.get((r, c), "-")
            if r in OVERHANG_ROWS and cell == "-":
                continue
            ax.add_patch(plt.Circle((i, y), 0.08, facecolor="white", edgecolor="#888888",
                                     linewidth=0.6, zorder=2))
            if cell == "X":
                # Screw hole: the strip there is physically unusable (a screw
                # through the board fixes it to its mount), marked with a cross.
                d = 0.14
                ax.plot([i - d, i + d], [y - d, y + d], color="#c0392b", linewidth=1.6, zorder=6)
                ax.plot([i - d, i + d], [y + d, y - d], color="#c0392b", linewidth=1.6, zorder=6)

    # --- cut marks ---
    for r in BOARD_ROWS:
        y = row_y(r, row_labels)
        for i, c in enumerate(col_names):
            if grid.get((r, c)) == "//":
                ax.add_patch(Rectangle((i - 0.16, y - 0.16), 0.32, 0.32, facecolor="white",
                                        edgecolor="none", zorder=3))
                ax.plot([i, i], [y - 0.16, y + 0.16], color="red", linewidth=1.4, zorder=6)

    # --- components (2 legs, same code) ---
    for code in sorted(COMPONENTS):
        hits = find_pair(grid, code, row_labels, col_names)
        if len(hits) != 2:
            continue
        (r1, c1), (r2, c2) = hits
        x1, y1 = col_names.index(c1), row_y(r1, row_labels)
        x2, y2 = col_names.index(c2), row_y(r2, row_labels)
        if code.startswith("R"):
            draw_zigzag(ax, x1, y1, x2, y2)
            dx, dy = x2 - x1, y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
            px, py = -dy / length, dx / length
            lx, ly = (x1 + x2) / 2 + px * 0.32, (y1 + y2) / 2 + py * 0.32
            ax.text(lx, ly, code, ha="center", va="center", fontsize=8, zorder=7)
        elif code == "C1-2":
            draw_capacitor(ax, x1, y1, y2, polarized=True)
            ax.text(x1 + 0.32, y1 + 0.3, "C1-2 220uF+100nF", ha="left", va="bottom",
                    fontsize=7, zorder=7)
        elif code == "TVS":
            draw_diode(ax, x1, y1, y2)
            ax.text(x1 + 0.3, y2 - 0.3, "TVS P6KE18A", ha="left", va="top", fontsize=7, zorder=7)

    # --- intra-board wires fN ---
    wire_codes = sorted({v for v in grid.values() if v.startswith("f") and v[1:].isdigit()})
    wire_spans = {}
    for code in wire_codes:
        hits = find_pair(grid, code, row_labels, col_names)
        if len(hits) != 2:
            continue
        (r1, c1), (r2, c2) = hits
        x1, y1 = col_names.index(c1), row_y(r1, row_labels)
        x2, y2 = col_names.index(c2), row_y(r2, row_labels)
        wire_spans[code] = (x1, y1, x2, y2, r1, r2)

    # Wires sharing a column get spread out with a small constant x-offset instead of
    # drawing exactly on top of each other.
    col_groups = {}
    for code, (x1, y1, x2, y2, r1, r2) in wire_spans.items():
        col = x1 if x1 == x2 else None
        col_groups.setdefault(col, []).append(code)
    offsets = {}
    for col, codes in col_groups.items():
        codes = sorted(codes)
        k = len(codes)
        for i, code in enumerate(codes):
            offsets[code] = 0.0 if col is None or k == 1 else (i - (k - 1) / 2) * 0.17

    # f4 (Col08) crosses R3's leg anchored on the same column at row06 - nudge it
    # aside by the same step used for grouped wires so the two stay distinguishable.
    offsets["f4"] = 0.17

    for code, (x1, y1, x2, y2, r1, r2) in wire_spans.items():
        color = WIRE_COLOR.get(code, "#2266cc")
        draw_wire(ax, x1, y1, y2, offsets.get(code, 0.0), color)

    # --- external wires (single occurrence on-board, one lead running off-board) ---
    # Real DRV8825 pins M1/M2 (Col12) share their text with the motor phase wires
    # (Col16) on purpose (documented homonymy) - exclude the real-pin columns so the
    # search only matches the actual wire's anchor.
    left_x = -2.3
    right_x = ncols + 1.3
    for code in sorted(EXTERNAL):
        hits = find_pair(grid, code, row_labels, col_names, exclude_cols=REAL_PIN_COLUMNS)
        if len(hits) != 1:
            continue
        (r, c) = hits[0]
        x, y = col_names.index(c), row_y(r, row_labels)
        color = EXTERNAL_COLOR.get(code, "black")
        slant = 0.5
        if code in ("ASDA", "ASCL", "AVCC", "AGND"):
            ax.plot([x, left_x], [y, y - slant], color=color, linewidth=1.4, zorder=4)
            ax.text(left_x - 0.15, y - slant, code, ha="right", va="center", fontsize=7.5,
                    color=color, zorder=7)
        else:  # M1-M4, T12V, TGND
            ax.plot([x, right_x], [y, y + slant], color=color, linewidth=1.4, zorder=4)
            ax.text(right_x + 0.15, y + slant, code, ha="left", va="center", fontsize=7.5,
                    color=color, zorder=7)

    # --- real pin labels ---
    for r in row_labels:
        y = row_y(r, row_labels)
        for c in col_names:
            cell = grid.get((r, c), "-")
            if cell in ("-", "//"):
                continue
            if c not in REAL_PIN_COLUMNS:
                continue
            x = col_names.index(c)
            ax.add_patch(plt.Circle((x, y), 0.09, facecolor="black", edgecolor="black", zorder=6))
            if c == "Col00":
                ax.text(x, y + 0.28, cell, ha="center", va="bottom", fontsize=6.5, zorder=7)
            elif c == "Col06":
                ax.text(x, y - 0.28, cell, ha="center", va="top", fontsize=6.5, zorder=7)
            elif c == "Col10":
                ax.text(x - 0.28, y, cell, ha="right", va="center", fontsize=6.5, zorder=7)
            elif c == "Col15":
                ax.text(x + 0.28, y, cell, ha="left", va="center", fontsize=6.5, zorder=7)

    # --- component footprints (dashed outlines, symbolic only) ---
    nano_top = row_y("R-3", row_labels)
    nano_bottom = row_y("R11", row_labels)
    ax.add_patch(Rectangle((NANO_COLS[0] - 0.5, nano_bottom - 0.5), NANO_COLS[1] - NANO_COLS[0] + 1,
                            nano_top - nano_bottom + 1, fill=False, edgecolor="#008080",
                            linestyle="--", linewidth=1.4, zorder=0))
    ax.text((NANO_COLS[0] + NANO_COLS[1]) / 2, nano_top + 1.3, "Arduino Nano", ha="center",
            fontsize=10, color="#008080")

    drv_top = row_y("R00", row_labels)
    drv_bottom = row_y("R07", row_labels)
    ax.add_patch(Rectangle((DRV_COLS[0] - 0.5, drv_bottom - 0.5), DRV_COLS[1] - DRV_COLS[0] + 1,
                            drv_top - drv_bottom + 1, fill=False, edgecolor="#800080",
                            linestyle="--", linewidth=1.4, zorder=0))
    ax.text((DRV_COLS[0] + DRV_COLS[1]) / 2, drv_bottom - 0.8, "DRV8825", ha="center",
            fontsize=10, color="#800080")

    # --- board outline (physical stripboard) ---
    board_top = row_y("R00", row_labels)
    board_bottom = row_y(BOARD_ROWS[-1], row_labels)
    ax.add_patch(Rectangle((-0.5, board_bottom - 0.5), ncols, board_top - board_bottom + 1,
                            fill=False, edgecolor="#1a4fa0", linewidth=2.2, zorder=0))

    board_w_mm = round((ncols - 1) * 2.54)
    board_h_mm = round((len(BOARD_ROWS) - 1) * 2.54)
    fig.suptitle(f"Stripboard {board_w_mm}x{board_h_mm}mm ({ncols}x{len(BOARD_ROWS)} trous, pas 2.54mm)",
                 fontsize=13, color="#1a4fa0")

    ax.set_xlim(-4.8, ncols + 3.2)
    ax.set_ylim(row_y(row_labels[-1], row_labels) - 2, nano_top + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
