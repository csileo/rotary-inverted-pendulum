"""Print a Stable-Baselines3 checkpoint's saved observation space, and the
`--frame-stack` value that reproduces it against `frame_stack.py`'s
`frame_dim=6` (pendulum raw observation), without needing an env/hardware.

Requires stable_baselines3 (see requirements.txt at the repo root).

Usage:
    python tools/checkpoints/inspect_sb3_obs_space.py <checkpoint.zip> [...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.save_util import load_from_zip_file

FRAME_DIM = 6


def inspect(zip_path: Path) -> None:
    data, _, _ = load_from_zip_file(str(zip_path), device="cpu", load_data=True)
    obs_space = data["observation_space"]
    shape = obs_space.shape
    n = shape[0] if shape else None
    frame_stack = n / FRAME_DIM if n else None
    note = ""
    if frame_stack is not None and frame_stack != int(frame_stack):
        note = "  (not a multiple of frame_dim=6 — unexpected obs construction)"
    print(f"{zip_path.name}: obs shape {shape}  ->  --frame-stack "
          f"{int(frame_stack) if frame_stack and frame_stack == int(frame_stack) else frame_stack}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", nargs="+", type=Path)
    args = parser.parse_args()
    for z in args.zips:
        inspect(z)


if __name__ == "__main__":
    main()
