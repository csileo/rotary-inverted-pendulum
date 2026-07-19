"""Summarise a run_policy.py --log trajectory for the models/README.md table.

Reuses analyze_run.py's pendulum_stats() (time-near-upright fractions,
longest sustained window) without requiring the motor_target_rad key that
fit_motor_lag() needs — some logs (e.g. plain run_policy.py validation runs,
as opposed to sysid recordings) don't have it, and analyze_run.py's main()
crashes on those.

Usage:
    python tools/validation_summary.py RotaryInvertedPendulum-python/src/rl/logs/policy_working_balance.npz
    python tools/validation_summary.py logs/a.npz logs/b.npz --near-top-threshold 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RL_DIR = Path(__file__).resolve().parent.parent / "RotaryInvertedPendulum-python" / "src" / "rl"
sys.path.insert(0, str(RL_DIR))
from analyze_run import pendulum_stats  # noqa: E402 (needs RL_DIR on sys.path first)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--near-top-threshold", type=int, default=15,
                   help="degrees from upright counted as 'balanced' for the "
                        "headline fraction (default 15, matches analyze_run.py)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    for path in args.logs:
        d = dict(np.load(path, allow_pickle=True))
        n = len(d["time_us"])
        control_freq = float(d["control_freq_hz"])
        dt_s = 1.0 / control_freq
        duration_s = n * dt_s

        pen = pendulum_stats(d["pendulum_pos_rad"].astype(np.float64), dt_s)
        frac = pen[f"upright_lt_{args.near_top_threshold}deg_frac"]
        longest = pen[f"longest_within_{args.near_top_threshold}deg_s"]

        print(f"\n=== {path.name} ({duration_s:.1f}s @ {control_freq:.0f} Hz) ===")
        print(f"  policy: {d.get('policy_path', 'unknown')}")
        print(f"  balance (< {args.near_top_threshold} deg): "
              f"{frac*100:.1f}% of run, longest continuous window {longest:.1f}s")
        print(f"  first catch (< 15 deg from upright): {pen['first_catch_s']:.2f}s")
        print(f"  README-ready: ~{frac*100:.1f}% balance over {duration_s:.0f}s "
              f"({longest:.1f}s longest continuous hold)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
