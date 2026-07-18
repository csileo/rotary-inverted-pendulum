"""Compare policies on balance-phase smoothness, not total episode reward.

`run_policy.py`'s live reward proxy (and the training reward it mirrors) is
dominated by swing-up variance — a slow or lucky swing-up swamps whatever
happens once the pendulum is near vertical. That makes total reward a poor
signal for judging chattering during balance, which is a purely local
phenomenon: sign-flipping motor velocity while the pendulum sits near the top.

This script isolates the "near top" steps (|pendulum_pos_rad| close to pi,
matching the sim's upright convention) and reports smoothness metrics on that
subset only, for one or more trajectory logs recorded via
`run_policy.py --log`.

Usage:
    python analyze_chattering.py logs/before.npz logs/after.npz
    python analyze_chattering.py --near-top-threshold 0.2 logs/*.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def analyze(path: Path, near_top_threshold: float) -> dict:
    d = np.load(path)
    pen_pos = d["pendulum_pos_rad"].astype(np.float64)
    motor_vel = d["motor_vel_rad_s"].astype(np.float64)
    action = d["action"].astype(np.float64)
    control_freq_hz = float(d["control_freq_hz"])
    dt = 1.0 / control_freq_hz

    # pendulum_pos_rad is an UNWRAPPED multi-revolution angle (firmware
    # tracks full rotations) — must go through wrap_pi(phi - pi), not the
    # naive abs(abs(phi) - pi), which silently breaks past one revolution
    # (e.g. a pendulum that swings through the top and keeps spinning).
    theta = _wrap_pi(pen_pos - math.pi)
    near_top = np.abs(theta) < near_top_threshold

    n_total = len(pen_pos)
    n_near_top = int(near_top.sum())

    # Longest unbroken stretch spent near the top — a wobble that leaves the
    # band and comes back still counts as an interruption, so this rewards
    # genuinely settled balance over a noisy in-and-out signal.
    longest_run = 0
    run = 0
    for flag in near_top:
        run = run + 1 if flag else 0
        longest_run = max(longest_run, run)

    result = {
        "file": path.name,
        "n_steps": n_total,
        "balance_fraction": n_near_top / n_total if n_total else 0.0,
        "longest_balance_run_s": longest_run * dt,
    }

    if n_near_top < 2:
        result.update(
            motor_vel_std=float("nan"),
            motor_jerk_rms=float("nan"),
            action_rate_rms=float("nan"),
            angle_rms_error_deg=float("nan"),
        )
        return result

    vel_bal = motor_vel[near_top]
    action_bal = action[near_top]
    # Deviation from the exact vertical, in degrees (wrap-safe).
    angle_err = np.abs(theta[near_top])

    result.update(
        motor_vel_std=float(np.std(vel_bal)),
        motor_jerk_rms=float(np.sqrt(np.mean(np.diff(vel_bal) ** 2)) / dt),
        action_rate_rms=float(np.sqrt(np.mean(np.diff(action_bal) ** 2))),
        angle_rms_error_deg=float(np.degrees(np.sqrt(np.mean(angle_err ** 2)))),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", type=Path, help="one or more .npz logs from run_policy.py --log")
    p.add_argument("--near-top-threshold", type=float, default=0.3,
                   help="rad from vertical (pi) counted as 'balancing' (default: 0.3)")
    args = p.parse_args(argv)

    rows = [analyze(path, args.near_top_threshold) for path in args.logs]

    header = (
        f"{'file':<40} {'balance %':>10} {'longest run (s)':>16} "
        f"{'motor_vel std':>14} {'jerk rms':>10} {'action-rate rms':>16} {'angle rms (deg)':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['file']:<40} {100 * r['balance_fraction']:>9.1f}% {r['longest_balance_run_s']:>16.2f} "
            f"{r['motor_vel_std']:>14.3f} {r['motor_jerk_rms']:>10.1f} "
            f"{r['action_rate_rms']:>16.4f} {r['angle_rms_error_deg']:>16.2f}"
        )

    print(
        "\nLower motor_vel std / jerk rms / action-rate rms = less chattering. "
        "Higher balance % and longest run = more time spent settled near vertical."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
