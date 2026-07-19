"""Free-swing dynamics probe — compare real pendulum decay against sim.

Captures the pendulum's intrinsic dynamics (gravity, friction, inertia)
with the motor *held fixed by hand* so the only forces are gravity and
joint friction. Compares fitted period and decay constant between real
and sim. This is the cleanest possible sim-to-real sanity check on the
pendulum model — no policy, no controller, no closed-loop coupling.

Procedure (operator-driven):
  1. Disengage motor. Operator holds the motor arm steady by hand and
     lifts the pendulum to ~90° from hanging.
  2. Script records pendulum angle while operator releases. Records for
     ~8 seconds.
  3. Script replays the same initial pendulum state in sim with motor
     held fixed via the PD position actuator at qpos=0.
  4. Prints comparison: period, decay constant, ratios. Saves a
     side-by-side plot.

Usage:
    python freeswing_probe.py --port /dev/cu.usbserial-1130
"""
import argparse
import math
import sys
import time

import mujoco
import numpy as np

from lowlevel_client import LowLevelClient
from pendulum_env import RotaryInvertedPendulumEnv


def record_real(port: str, duration_s: float, sample_hz: float) -> dict:
    dt = 1.0 / sample_hz
    n = int(duration_s * sample_hz)
    t_log = np.zeros(n)
    motor_log = np.zeros(n)
    pen_log = np.zeros(n)
    mvel_log = np.zeros(n)
    pvel_log = np.zeros(n)

    with LowLevelClient(port) as client:
        if not client.wait_until_ready():
            print("Arduino not responding"); sys.exit(1)
        client.disengage_motor()
        time.sleep(0.3)

        print()
        print("=" * 64)
        print("Hold the motor arm STILL by hand, lift the pendulum to ~90°")
        print("from its hanging position, hold it there. Press Enter when ready.")
        print("=" * 64)
        input()
        print("Recording for", duration_s, "seconds. RELEASE the pendulum now")
        print("(but keep the motor arm fixed). 3..."); time.sleep(1)
        print("2..."); time.sleep(1)
        print("1..."); time.sleep(1)
        print("GO.")

        t_start = time.monotonic()
        next_tick = t_start
        for i in range(n):
            s = client.get_state()
            t_log[i] = (time.monotonic() - t_start)
            motor_log[i] = -s.motor_pos_rad
            pen_log[i]   = -s.pendulum_pos_rad
            mvel_log[i]  = -s.motor_vel_rad_s
            pvel_log[i]  = -s.pendulum_vel_rad_s
            next_tick += dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    return dict(t=t_log, motor=motor_log, pen=pen_log,
                mvel=mvel_log, pvel=pvel_log)


def replay_sim(real_data: dict, sample_hz: float, control_freq_hz: float = 200.0) -> dict:
    """Replay free-swing in sim from the same initial conditions.

    The firmware's pendulum-angle accumulator has an arbitrary offset
    (whatever raw value the AS5600 read at Arduino boot). Sim's hanging
    is at qpos=0 by construction. We bridge the two by computing the
    release angle *relative to real's settled equilibrium* and using
    that as sim's qpos.
    """
    n = len(real_data["t"])
    pen_eq_real = float(np.median(real_data["pen"][-int(0.5 * sample_hz):]))
    release_angle = float(real_data["pen"][0]) - pen_eq_real
    pvel0 = float(real_data["pvel"][0])
    motor0 = float(real_data["motor"][0])

    env = RotaryInvertedPendulumEnv(
        control_freq_hz=control_freq_hz,
        domain_randomization=False,
        episode_length_s=real_data["t"][-1] + 0.5,
    )
    env.reset(seed=0)
    env.data.qpos[env._motor_qpos_addr] = motor0
    env.data.qpos[env._pen_qpos_addr]   = release_angle
    env.data.qvel[env._motor_qvel_addr] = 0.0
    env.data.qvel[env._pen_qvel_addr]   = pvel0
    env._motor_target = motor0
    env._motor_vel    = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Step sim at control_freq_hz; sample at sample_hz.
    sim_dt = 1.0 / control_freq_hz
    sample_dt = 1.0 / sample_hz
    sim_t = np.zeros(n); sim_pen = np.zeros(n); sim_pvel = np.zeros(n)
    sample_idx = 0; next_sample_t = 0.0
    t = 0.0
    while sample_idx < n:
        # Action = 0 → no accel command, motor target stays put.
        env.step(np.array([0.0], dtype=np.float32))
        t += sim_dt
        while sample_idx < n and t >= next_sample_t:
            sim_t[sample_idx] = next_sample_t
            sim_pen[sample_idx] = float(env.data.qpos[env._pen_qpos_addr])
            sim_pvel[sample_idx] = float(env.data.qvel[env._pen_qvel_addr])
            sample_idx += 1
            next_sample_t += sample_dt

    return dict(t=sim_t, pen=sim_pen, pvel=sim_pvel)


def fit_decay_envelope(t: np.ndarray, theta: np.ndarray, theta_eq: float) -> dict:
    """Fit an exponentially-decaying envelope to peaks of (theta - theta_eq).
    Returns period_s, decay_constant_s_inv (negative slope of log|peak|).
    """
    y = theta - theta_eq
    # Find sign changes → half-periods, peaks in between
    signs = np.sign(y)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) < 4:
        return dict(period_s=float('nan'), decay_const=float('nan'),
                    peak_times=[], peak_amps=[])
    # Peaks: between consecutive crossings, find argmax|y|
    peak_t, peak_amp = [], []
    for k in range(len(crossings) - 1):
        a, b = crossings[k] + 1, crossings[k + 1] + 1
        j = a + int(np.argmax(np.abs(y[a:b])))
        peak_t.append(t[j]); peak_amp.append(abs(y[j]))
    peak_t = np.array(peak_t); peak_amp = np.array(peak_amp)
    # Period: 2 × median half-period
    half_periods = np.diff(peak_t)
    period = 2 * np.median(half_periods)
    # Decay: linear fit of log|peak| vs t
    mask = peak_amp > 1e-4
    if mask.sum() < 3:
        return dict(period_s=period, decay_const=float('nan'),
                    peak_times=peak_t, peak_amps=peak_amp)
    coef = np.polyfit(peak_t[mask], np.log(peak_amp[mask]), 1)
    decay = -float(coef[0])  # positive = decaying
    return dict(period_s=period, decay_const=decay,
                peak_times=peak_t, peak_amps=peak_amp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--sample-hz", type=float, default=100.0)
    p.add_argument("--save", default="/tmp/freeswing.npz",
                   help="path to save real + sim traces")
    p.add_argument("--plot", default="/tmp/freeswing.png",
                   help="path to save comparison plot")
    args = p.parse_args()

    real = record_real(args.port, args.duration_s, args.sample_hz)
    print()
    print("Replaying in sim...")
    sim = replay_sim(real, args.sample_hz)

    # Real's "hanging" is wherever the firmware's encoder accumulator
    # happens to converge — its zero is arbitrary. Sim's hanging is at
    # qpos=0 by env construction. So we use observed real equilibrium
    # but force sim's equilibrium to 0.
    theta_eq_real = float(np.median(real["pen"][-int(0.5 * args.sample_hz):]))
    theta_eq_sim  = 0.0

    print()
    print(f"Real: pen_0 = {real['pen'][0]:+.3f} rad,  pen_eq = {theta_eq_real:+.3f}")
    print(f"Sim:  pen_0 = {sim['pen'][0]:+.3f} rad,   pen_eq = {theta_eq_sim:+.3f}")
    print()

    fit_real = fit_decay_envelope(real["t"], real["pen"], theta_eq_real)
    fit_sim  = fit_decay_envelope(sim["t"],  sim["pen"],  theta_eq_sim)

    def fmt(f): return f"{f:.4f}" if not math.isnan(f) else "(nan)"
    print(f"  {'metric':<25s}  {'real':>12s}  {'sim':>12s}  {'sim − real':>12s}")
    print(f"  {'period (s)':<25s}  {fmt(fit_real['period_s']):>12s}  "
          f"{fmt(fit_sim['period_s']):>12s}  "
          f"{fmt(fit_sim['period_s'] - fit_real['period_s']):>12s}")
    print(f"  {'decay const (1/s)':<25s}  {fmt(fit_real['decay_const']):>12s}  "
          f"{fmt(fit_sim['decay_const']):>12s}  "
          f"{fmt(fit_sim['decay_const'] - fit_real['decay_const']):>12s}")
    print()
    print("Interpretation:")
    print("  - period mismatch → inertia or COM is wrong")
    print("  - decay mismatch → friction is wrong (sim > real ⇒ sim overdamped)")

    np.savez(args.save, real_t=real["t"], real_pen=real["pen"],
             real_motor=real["motor"], real_mvel=real["mvel"], real_pvel=real["pvel"],
             sim_t=sim["t"], sim_pen=sim["pen"], sim_pvel=sim["pvel"])
    print(f"\nSaved traces to {args.save}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(real["t"], real["pen"] - theta_eq_real, label="real", lw=1.0)
        axes[0].plot(sim["t"],  sim["pen"]  - theta_eq_sim,  label="sim",  lw=1.0, alpha=0.85)
        axes[0].set_ylabel("pendulum angle from equilibrium (rad)")
        axes[0].axhline(0, color="k", lw=0.3); axes[0].legend(); axes[0].grid(alpha=0.3)
        if len(fit_real["peak_times"]):
            axes[1].semilogy(fit_real["peak_times"], fit_real["peak_amps"], "o-",
                              label=f"real peaks  (τ={fit_real['decay_const']:.3f} 1/s)")
        if len(fit_sim["peak_times"]):
            axes[1].semilogy(fit_sim["peak_times"], fit_sim["peak_amps"], "s--",
                              label=f"sim peaks   (τ={fit_sim['decay_const']:.3f} 1/s)")
        axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("|peak amplitude| (rad, log)")
        axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
        plt.tight_layout(); plt.savefig(args.plot, dpi=120)
        print(f"Saved plot to {args.plot}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
