"""Analyse a real-hardware trajectory log produced by `run_policy.py --log`.

Extracts:
- Effective motor first-order lag (tau) by fitting motor_actual to a
  delayed-and-lagged version of motor_target.
- Pure transport delay between target and motor response.
- Dead-zone / stiction signature: residual after lag fit.
- Pendulum trajectory statistics (time near upright, oscillation amplitude).

Usage:
    python analyze_run.py /tmp/run_2026-05-01.npz
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize_scalar


def load(path: str) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    return d


def _first_order(target: np.ndarray, tau_steps: float) -> np.ndarray:
    """Apply a first-order lag with time constant tau_steps to a target series."""
    out = np.zeros_like(target)
    out[0] = target[0]
    if tau_steps <= 0:
        return target.copy()
    alpha = 1.0 - math.exp(-1.0 / tau_steps)
    for i in range(1, len(target)):
        out[i] = out[i - 1] + alpha * (target[i - 1] - out[i - 1])
    return out


def fit_motor_lag(target: np.ndarray, actual: np.ndarray, dt_s: float) -> dict:
    """Fit transport delay (integer steps) and first-order tau (continuous)."""
    # 1. Estimate pure transport delay via cross-correlation peak.
    # Centre both signals so DC component doesn't dominate.
    t = target - target.mean()
    a = actual - actual.mean()
    if np.std(t) < 1e-6 or np.std(a) < 1e-6:
        return {"delay_steps": 0, "tau_s": 0.0, "rmse_rad": float("nan"),
                "note": "constant target or actual; nothing to fit"}
    corr = correlate(a, t, mode="full")
    centre = len(t) - 1
    # Search a reasonable window (-5 .. +20 steps)
    window = slice(max(0, centre - 5), min(len(corr), centre + 20))
    delay_steps = int(np.argmax(corr[window]) + window.start - centre)
    delay_steps = max(0, delay_steps)

    # 2. Fit tau_steps that minimises residual RMSE.
    def loss(tau_steps: float) -> float:
        if tau_steps < 0:
            return 1e9
        # Apply transport delay first by shifting target right by delay_steps
        shifted = np.concatenate([np.full(delay_steps, target[0]), target[:len(target) - delay_steps]])
        predicted = _first_order(shifted, tau_steps)
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))

    res = minimize_scalar(loss, bounds=(0.0, 10.0), method="bounded",
                          options={"xatol": 0.01})
    tau_steps = float(res.x)
    rmse = float(res.fun)

    return {
        "delay_steps": delay_steps,
        "delay_s": delay_steps * dt_s,
        "tau_steps": tau_steps,
        "tau_s": tau_steps * dt_s,
        "rmse_rad": rmse,
    }


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest contiguous True segment in `mask`."""
    if len(mask) == 0:
        return 0
    longest = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def pendulum_stats(phi: np.ndarray, dt_s: float) -> dict:
    """Compute pendulum trajectory statistics.

    Distinguishes "transient near-upright" from "sustained balance" by
    reporting the longest contiguous window within ±15 / 30 / 60 degrees,
    in addition to the simple time fractions. A policy that swings through
    upright many times can score high on the fraction but have a small
    longest-window — that's the failure mode we want to catch.
    """
    # theta = phi - pi  (theta=0 at upright). The +math.pi/2*math.pi trick
    # keeps it bounded into [-pi, pi].
    theta = ((phi - math.pi + math.pi) % (2 * math.pi)) - math.pi

    out = {}
    for thr_deg in (15, 30, 60):
        thr = math.radians(thr_deg)
        mask = np.abs(theta) < thr
        out[f"upright_lt_{thr_deg}deg_frac"] = float(np.mean(mask))
        out[f"longest_within_{thr_deg}deg_s"] = _longest_run(mask) * dt_s

    # Time-to-first-catch: first sample within ±15 deg of upright.
    catches = np.where(np.abs(theta) < math.radians(15))[0]
    out["first_catch_s"] = float(catches[0]) * dt_s if len(catches) > 0 else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyse a run_policy trajectory log")
    p.add_argument("log", help="path to .npz produced by run_policy.py --log")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    d = load(args.log)
    n = len(d["time_us"])
    control_freq = float(d["control_freq_hz"])
    dt_s = 1.0 / control_freq

    print(f"Run: {d.get('policy_path', 'unknown')}")
    print(f"Steps: {n}  duration: {n * dt_s:.2f}s  control_dt: {dt_s*1e3:.1f} ms")
    print()

    # Sample-rate sanity check
    t_us = d["time_us"].astype(np.int64)
    dt_us = np.diff(t_us)
    print(f"Sample dt_us: median={float(np.median(dt_us)):.0f}, "
          f"mean={float(np.mean(dt_us)):.1f}, std={float(np.std(dt_us)):.1f}")
    print()

    print("--- Motor lag fit (target -> actual) ---")
    lag = fit_motor_lag(
        d["motor_target_rad"].astype(np.float64),
        d["motor_pos_rad"].astype(np.float64),
        dt_s,
    )
    for k, v in lag.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print()

    print("--- Pendulum stats ---")
    pen = pendulum_stats(d["pendulum_pos_rad"].astype(np.float64), dt_s)
    for k, v in pen.items():
        print(f"  {k}: {v:.3f}")
    print()

    # Ranges
    print("--- Signal ranges ---")
    print(f"  motor_pos_rad   range: [{float(d['motor_pos_rad'].min()):+.3f}, {float(d['motor_pos_rad'].max()):+.3f}]")
    print(f"  motor_target    range: [{float(d['motor_target_rad'].min()):+.3f}, {float(d['motor_target_rad'].max()):+.3f}]")
    print(f"  pendulum_pos    range: [{float(d['pendulum_pos_rad'].min()):+.3f}, {float(d['pendulum_pos_rad'].max()):+.3f}]")
    print(f"  motor_vel       range: [{float(d['motor_vel_rad_s'].min()):+.2f}, {float(d['motor_vel_rad_s'].max()):+.2f}] rad/s")
    print(f"  pendulum_vel    range: [{float(d['pendulum_vel_rad_s'].min()):+.2f}, {float(d['pendulum_vel_rad_s'].max()):+.2f}] rad/s")
    print(f"  action          range: [{float(d['action'].min()):+.3f}, {float(d['action'].max()):+.3f}]")
    print()

    # Sysid comparison
    print("--- vs sysid_params.json motor step fit ---")
    print(f"  sysid step rise time (95%):     0.064 s   ~tau ≈ 21 ms")
    print(f"  policy-driven tau fit:          {lag.get('tau_s', float('nan'))*1000:.1f} ms")
    print(f"  policy-driven transport delay:  {lag.get('delay_s', float('nan'))*1000:.1f} ms ({lag.get('delay_steps', 0)} steps)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
