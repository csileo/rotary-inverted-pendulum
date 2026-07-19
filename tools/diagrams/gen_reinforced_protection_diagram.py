"""Finalize system-without-batteries-reinforced-protection.jpg from its draft.

system-without-batteries-reinforced-protection0.jpg is a hand-built draft
(source .drawio lost) that already shows the reinforced-protection additions
on top of the system-without-batteries.jpg composition: 3 series resistors
(R1/R2/R3) spliced into the D2/D5/D9 signal lines, a 220uF+100nF cap pair and
a P6KE18A TVS diode near VMOT/GND, and a USB isolator spliced into the Nano's
USB cable - all matching stripboard_matrix.csv (R1, R2, R3, C1-2, TVS). It was
checked once against that CSV and against system-without-batteries.drawio's
current wire colors; both matched, so the draft is simply promoted to the
final filename.

Usage:
    python tools/diagrams/gen_reinforced_protection_diagram.py [--draft PATH] [--out PATH]
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGRAMS = REPO_ROOT / "diagrams"
DEFAULT_DRAFT = DIAGRAMS / "system-without-batteries-reinforced-protection0.jpg"
DEFAULT_OUT = DIAGRAMS / "system-without-batteries-reinforced-protection.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default=str(DEFAULT_DRAFT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    shutil.copyfile(args.draft, args.out)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
