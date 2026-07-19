"""Copy a trained SB3 checkpoint (+ optional replay buffer, + optional
validation log(s)) into this repo's `models/` and `logs/` directories under
a new, short base name.

Generalises the ad-hoc copy done by hand for the demo models (working
balance/partial balance/fails to swing up): pick a run's checkpoint, its
matching replay buffer, and its hardware validation log(s), rename them
consistently, and drop them where `models/README.md` and
`run_policy.py --policy` expect them.

Sanity checks it does NOT do for you: it won't verify a buffer actually
matches the checkpoint's episode (see the working-balance vs. partial-balance
case in models/README.md, where the only available buffer was stale) beyond
a loose mtime comparison — read the printed warning and check the source run
directory yourself if it fires.

Requires stable_baselines3 (see RotaryInvertedPendulum-python/src/rl/requirements.txt).

Usage:
    python tools/checkpoints/save_model_bundle.py \\
        --model runs/some_run/checkpoint.zip --name policy_something

    python tools/checkpoints/save_model_bundle.py \\
        --model runs/finetune_curriculum9/checkpoint_ep050.zip \\
        --buffer runs/finetune_curriculum9/replay_buffer_ep050.pkl \\
        --log logs/run1.npz logs/run2.npz \\
        --name policy_curriculum9_ep050
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from stable_baselines3.common.save_util import load_from_zip_file

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_DIR = "RotaryInvertedPendulum-python/src/rl"
FRAME_DIM = 6

MTIME_WARN_THRESHOLD_S = 60.0


def _print_obs_space(model_path: Path) -> None:
    try:
        data, _, _ = load_from_zip_file(str(model_path), device="cpu", load_data=True)
    except Exception as e:
        print(f"  (could not inspect obs space: {e})")
        return
    shape = data["observation_space"].shape
    n = shape[0] if shape else None
    frame_stack = n / FRAME_DIM if n else None
    if frame_stack and frame_stack == int(frame_stack):
        print(f"  obs shape {shape}  ->  run with --frame-stack {int(frame_stack)}")
    else:
        print(f"  obs shape {shape}  ->  not a multiple of frame_dim={FRAME_DIM}, check manually")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="source checkpoint .zip")
    parser.add_argument("--buffer", type=Path, default=None,
                         help="source replay_buffer .pkl (optional)")
    parser.add_argument("--log", type=Path, nargs="*", default=[],
                         help="source validation .npz log(s) (optional, one or more)")
    parser.add_argument("--name", required=True,
                         help="new base name, e.g. policy_curriculum9_ep050")
    parser.add_argument("--target", type=Path, default=REPO_ROOT,
                         help="target repo root (default: this repo)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if args.buffer is not None and not args.buffer.exists():
        raise SystemExit(f"buffer not found: {args.buffer}")
    for log in args.log:
        if not log.exists():
            raise SystemExit(f"log not found: {log}")

    models_dir = args.target / RL_DIR / "models"
    logs_dir = args.target / RL_DIR / "logs"

    plan: list[tuple[Path, Path]] = [(args.model, models_dir / f"{args.name}.zip")]
    if args.buffer is not None:
        plan.append((args.buffer, models_dir / f"{args.name}_replay_buffer.pkl"))
    if len(args.log) == 1:
        plan.append((args.log[0], logs_dir / f"{args.name}.npz"))
    else:
        for i, log in enumerate(args.log, start=1):
            plan.append((log, logs_dir / f"{args.name}_{i}.npz"))

    for src, dst in plan:
        exists = " (OVERWRITES existing file)" if dst.exists() else ""
        print(f"{src} -> {dst.relative_to(args.target)}{exists}")

    if args.buffer is not None:
        dt = abs(args.model.stat().st_mtime - args.buffer.stat().st_mtime)
        if dt > MTIME_WARN_THRESHOLD_S:
            print(f"\nWARNING: model and buffer mtimes differ by {dt:.0f}s — "
                  f"they may not correspond to the same episode. Verify against "
                  f"the source run directory before relying on this buffer to "
                  f"resume fine-tuning.")

    if args.dry_run:
        print("\n(dry run — nothing copied)")
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in plan:
        shutil.copyfile(src, dst)

    print(f"\nCopied {len(plan)} file(s).")
    print(f"\n{args.name}.zip:")
    _print_obs_space(models_dir / f"{args.name}.zip")


if __name__ == "__main__":
    main()
