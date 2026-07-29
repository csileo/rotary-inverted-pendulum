"""Export a distilled student `.pt` checkpoint as a plain-NumPy `.npz`.

One-time conversion, meant to run on a machine that already has PyTorch
(the dev PC) — the output `.npz` needs nothing but NumPy to run, so
`numpy_student.py` can do inference on a Raspberry Pi (or anywhere else)
without ever installing torch/stable-baselines3. See docs/quantisation.md
for the on-device (Arduino Nano) int8 export path — this script is for the
opposite end: dropping the *host-side* PyTorch dependency, not shrinking
the model.

Usage:
    python export_weights_numpy.py \\
        --student models/distill_working_balance_h32_dagger/student.pt \\
        --out     models/distill_working_balance_h32_dagger/student_numpy.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from distill import StudentMLP, _extract_weights


def export(student_path: Path, out_path: Path) -> dict:
    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=True)
    model = StudentMLP(
        hidden=int(ckpt["hidden"]),
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    weights = _extract_weights(model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        **weights,
        hidden=np.int64(ckpt["hidden"]),
        obs_dim=np.int64(ckpt["obs_dim"]),
        act_dim=np.int64(ckpt["act_dim"]),
        val_mse=np.float64(ckpt.get("val_mse", float("nan"))),
    )
    print(f"[export] {student_path} -> {out_path} "
          f"({out_path.stat().st_size / 1e3:.1f} KB)")
    return weights


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", required=True, type=Path,
                   help="path to a student.pt produced by distill.py")
    p.add_argument("--out", required=True, type=Path,
                   help="output .npz path")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    export(args.student, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
