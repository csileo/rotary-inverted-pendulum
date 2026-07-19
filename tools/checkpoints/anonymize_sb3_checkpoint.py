"""Strip absolute local filesystem paths from a Stable-Baselines3 checkpoint
.zip's `data` JSON (e.g. `tensorboard_log`, which SB3 saves as whatever
absolute path was live on the training machine — leaks the username and
folder layout). Use scan_sb3_checkpoint_paths.py first to find what needs
replacing.

Rewrites only the `data` entry; every other zip member (policy.pth,
optimizer states, system_info.txt, ...) is copied through byte-for-byte,
same compression, same order.

No dependencies beyond the standard library.

Usage:
    python tools/checkpoints/anonymize_sb3_checkpoint.py <checkpoint.zip> \\
        --replace "C:\\Users\\me\\...\\runs\\some_run\\tb=runs/some_run/tb"
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def anonymize(zip_path: Path, replacements: dict[str, str]) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        contents = {i.filename: zf.read(i.filename) for i in infos}

    data = json.loads(contents["data"])
    changed = []
    for key, value in list(data.items()):
        if isinstance(value, str) and value in replacements:
            data[key] = replacements[value]
            changed.append((key, value, replacements[value]))
    if not changed:
        raise SystemExit(f"no matching value found to replace in {zip_path}")

    contents["data"] = json.dumps(data).encode("utf-8")

    tmp_path = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_path, "w") as zf_out:
        for info in infos:
            zf_out.writestr(info, contents[info.filename])
    tmp_path.replace(zip_path)

    for key, old, new in changed:
        print(f"{zip_path}: data['{key}'] {old!r} -> {new!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--replace", action="append", required=True,
                         help="old=new pair (old must match a top-level string value in `data` exactly)")
    args = parser.parse_args()

    replacements = {}
    for pair in args.replace:
        old, _, new = pair.partition("=")
        replacements[old] = new

    anonymize(args.zip_path, replacements)


if __name__ == "__main__":
    main()
