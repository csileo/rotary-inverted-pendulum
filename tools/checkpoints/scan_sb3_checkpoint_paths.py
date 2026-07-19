"""Scan a Stable-Baselines3 checkpoint .zip's `data` JSON for absolute
local filesystem paths (e.g. tensorboard_log), which routinely leak the
machine's username/folder layout since SB3 stores whatever `--run-name`
path was live at save time. Run this before committing a checkpoint into
models/ to make sure nothing local leaks into the public repo.

No dependencies beyond the standard library.

Usage:
    python tools/checkpoints/scan_sb3_checkpoint_paths.py <checkpoint.zip> [...]
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

SUSPECT_MARKERS = ("Users\\", "Users/", "/home/", ":\\", "csileo")


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(obj, str) and any(m in obj for m in SUSPECT_MARKERS):
        yield path, obj


def scan(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        data = json.loads(zf.read("data"))
    hits = list(_walk(data))
    print(f"=== {zip_path} ===")
    if not hits:
        print("  (no suspect path found)")
    for path, value in hits:
        print(f"  {path} => {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", nargs="+", type=Path)
    args = parser.parse_args()
    for z in args.zips:
        scan(z)


if __name__ == "__main__":
    main()
