"""Diagnose swing-up efficiency from a run_policy.py --log trajectory.

Segments a run into alternating swing-up / balance phases (with hysteresis,
so a single noisy step doesn't flip the state) and reports, per swing-up
phase: how long it took and how many direction reversals (pendulum velocity
sign changes) it needed before settling — a proxy for "how many pumps/swings"
a human would count watching the rig. A run that falls out of balance and
re-swings shows up as more than one swing-up phase.

Usage:
    python analyze_swingup.py logs/curriculum8_ep050_11.npz
    python analyze_swingup.py logs/curriculum8_ep050_*.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def segment(path: Path, near_top_threshold: float, min_balance_run_s: float,
            min_exit_run_s: float) -> dict:
    d = np.load(path)
    pen_pos = d["pendulum_pos_rad"].astype(np.float64)
    pen_vel = d["pendulum_vel_rad_s"].astype(np.float64)
    control_freq_hz = float(d["control_freq_hz"])
    dt = 1.0 / control_freq_hz
    n = len(pen_pos)

    # pendulum_pos_rad is an UNWRAPPED multi-revolution angle (the firmware
    # tracks full rotations, not just [-pi, pi]) — a pendulum that swings
    # through the top and keeps spinning can reach values far outside one
    # revolution. Distance-from-upright must go through wrap_pi(phi - pi)
    # (theta=0 at upright, matches pendulum_env.py's convention), not the
    # naive abs(abs(phi) - pi), which silently breaks past one revolution.
    theta = _wrap_pi(pen_pos - math.pi)
    near_top = np.abs(theta) < near_top_threshold
    min_run = max(1, int(round(min_balance_run_s * control_freq_hz)))
    min_exit = max(1, int(round(min_exit_run_s * control_freq_hz)))

    swingups = []  # list of dicts: start_idx, end_idx, n_reversals
    balances = []  # list of dicts: start_idx, end_idx

    state = "SWINGING"
    seg_start = 0
    consec_in = 0
    consec_out = 0

    for i in range(n):
        if near_top[i]:
            consec_in += 1
            consec_out = 0
        else:
            consec_out += 1
            consec_in = 0

        if state == "SWINGING" and consec_in >= min_run:
            balance_start = i - min_run + 1
            vel_seg = pen_vel[seg_start:balance_start]
            signs = np.sign(vel_seg)
            signs = signs[signs != 0]
            n_reversals = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
            swingups.append(dict(start_idx=seg_start, end_idx=balance_start,
                                  n_reversals=n_reversals))
            state = "BALANCED"
            seg_start = balance_start
            consec_in = 0
        elif state == "BALANCED" and consec_out >= min_exit:
            fall_start = i - min_exit + 1
            balances.append(dict(start_idx=seg_start, end_idx=fall_start))
            state = "SWINGING"
            seg_start = fall_start
            consec_out = 0

    # Close the trailing open segment.
    if state == "SWINGING" and seg_start < n:
        vel_seg = pen_vel[seg_start:n]
        signs = np.sign(vel_seg)
        signs = signs[signs != 0]
        n_reversals = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
        swingups.append(dict(start_idx=seg_start, end_idx=n, n_reversals=n_reversals,
                              incomplete=True))
    elif state == "BALANCED" and seg_start < n:
        balances.append(dict(start_idx=seg_start, end_idx=n, incomplete=True))

    return dict(file=path.name, dt=dt, n=n, swingups=swingups, balances=balances)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--near-top-threshold", type=float, default=0.3)
    p.add_argument("--min-balance-run-s", type=float, default=1.0,
                   help="consecutive seconds near top required to call it 'balanced' (default 1.0)")
    p.add_argument("--min-exit-run-s", type=float, default=0.3,
                   help="consecutive seconds outside the band required to call it 'fell' (default 0.3)")
    args = p.parse_args(argv)

    for path in args.logs:
        r = segment(path, args.near_top_threshold, args.min_balance_run_s, args.min_exit_run_s)
        dt = r["dt"]
        print(f"\n=== {r['file']} ({r['n']} steps, {r['n']*dt:.1f}s) ===")
        if not r["swingups"]:
            print("  (never left the balance band — started near top?)")
        for k, sw in enumerate(r["swingups"]):
            dur = (sw["end_idx"] - sw["start_idx"]) * dt
            tag = " [never stabilised]" if sw.get("incomplete") else ""
            print(f"  swing-up #{k+1}: t={sw['start_idx']*dt:.2f}s-{sw['end_idx']*dt:.2f}s "
                  f"({dur:.2f}s, {sw['n_reversals']} reversals){tag}")
        for k, bal in enumerate(r["balances"]):
            dur = (bal["end_idx"] - bal["start_idx"]) * dt
            tag = " [held to end]" if bal.get("incomplete") else " [fell]"
            print(f"  balance  #{k+1}: t={bal['start_idx']*dt:.2f}s-{bal['end_idx']*dt:.2f}s "
                  f"({dur:.2f}s){tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
