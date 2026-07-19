"""Isolate which DR dimension breaks a deployed policy.

For each DR dimension in pendulum_env, runs N episodes with ONLY that
dimension varying (others fixed at nominal). Reports solve rate per
condition.

Useful for understanding *why* a policy fails when full DR is applied
and which dimension is most worth ramping into the training
curriculum.

Usage:
    python eval_dr_sensitivity.py runs/<run>/best_model.zip
    python eval_dr_sensitivity.py runs/<run>/best_model.zip --n-episodes 20
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

import pendulum_env as pe
from pendulum_env import _wrap_pi, DR_MOTOR_ACCEL_RANGE_RAD_S2


def run_eval(model, label: str, *,
             n_episodes: int, episode_steps: int, control_freq: float,
             upright_threshold_rad: float,
             dr_accel=None, dr_delay=None, dr_dt_jitter=None,
             mass_frac=0.0, com_frac=0.0, friction_mult=None,
             dr_friction_loss=False) -> int:
    """Run n_episodes with a specific DR profile (others fixed at nominal)."""
    orig_mass = pe.DR_PENDULUM_MASS_FRAC
    orig_com = pe.DR_PENDULUM_COM_FRAC
    orig_friction = pe.DR_PENDULUM_FRICTION_MULT_RANGE
    orig_friction_loss = pe.DR_MOTOR_FRICTIONLOSS_RANGE_N_M

    pe.DR_PENDULUM_MASS_FRAC = mass_frac
    pe.DR_PENDULUM_COM_FRAC = com_frac
    pe.DR_PENDULUM_FRICTION_MULT_RANGE = friction_mult if friction_mult else (1.0, 1.0)
    pe.DR_MOTOR_FRICTIONLOSS_RANGE_N_M = orig_friction_loss if dr_friction_loss else (0.0, 0.0)

    try:
        env = pe.RotaryInvertedPendulumEnv(
            control_freq_hz=control_freq,
            domain_randomization=True,
            dr_motor_accel_range_rad_s2=dr_accel if dr_accel else (150.0, 150.0),
            dr_action_delay_steps_range=dr_delay if dr_delay else (0, 0),
            dr_control_dt_jitter_frac=dr_dt_jitter if dr_dt_jitter is not None else 0.0,
            episode_length_s=episode_steps / control_freq,
        )
        n_solved = n_hardstop = 0
        rewards = []
        upright_pcts = []
        hold_window = int(1.0 * control_freq)
        for i in range(n_episodes):
            obs, _ = env.reset(seed=1000 + i)
            ep_reward = 0.0
            upright_hist = []
            hit_limit = False
            for _ in range(episode_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                theta = _wrap_pi(float(info['phi']) - math.pi)
                upright_hist.append(abs(theta) < upright_threshold_rad)
                if abs(float(info['motor_pos'])) > math.radians(130):
                    hit_limit = True
                if term or trunc:
                    break
            held = sum(upright_hist[-hold_window:]) / max(1, hold_window)
            upright_pcts.append(held)
            rewards.append(ep_reward)
            if held >= 0.9 and not hit_limit:
                n_solved += 1
            if hit_limit:
                n_hardstop += 1

        print(f"  {label:40s}  solved={n_solved:2d}/{n_episodes}  "
              f"mean_upright={np.mean(upright_pcts)*100:5.1f}%  "
              f"hardstop={n_hardstop:2d}  reward={np.mean(rewards):+.0f}")
        return n_solved
    finally:
        pe.DR_PENDULUM_MASS_FRAC = orig_mass
        pe.DR_PENDULUM_COM_FRAC = orig_com
        pe.DR_PENDULUM_FRICTION_MULT_RANGE = orig_friction
        pe.DR_MOTOR_FRICTIONLOSS_RANGE_N_M = orig_friction_loss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", type=Path)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--episode-length-s", type=float, default=8.0)
    p.add_argument("--control-freq", type=float, default=35.0)
    p.add_argument("--upright-threshold-deg", type=float, default=15.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = SAC.load(str(args.policy), device=args.device)
    upright_threshold_rad = math.radians(args.upright_threshold_deg)
    episode_steps = int(args.episode_length_s * args.control_freq)

    def go(label, **kw):
        run_eval(model, label,
                 n_episodes=args.n_episodes, episode_steps=episode_steps,
                 control_freq=args.control_freq,
                 upright_threshold_rad=upright_threshold_rad, **kw)

    print(f"Policy: {args.policy}")
    print(f"N={args.n_episodes} eps × {args.episode_length_s:.1f}s @ {args.control_freq:.0f}Hz, "
          f"upright threshold ±{args.upright_threshold_deg:.0f}°")
    print()
    print(f"  {'Condition':40s}  Result")

    go("baseline (all fixed)")
    print("\n--- isolate each DR dim ---")
    go("action delay [1,1]", dr_delay=(1, 1))
    go("action delay [2,2]", dr_delay=(2, 2))
    go("action delay [3,3]", dr_delay=(3, 3))
    go("motor_accel [130,130]", dr_accel=(130, 130))
    go("motor_accel [170,170]", dr_accel=(170, 170))
    go("motor_accel [110,110]", dr_accel=(110, 110))
    go("motor_accel [190,190]", dr_accel=(190, 190))
    go("pendulum mass ±20%", mass_frac=0.20)
    go("pendulum COM ±10%", com_frac=0.10)
    go("pendulum friction ×[0.5, 2.0]", friction_mult=(0.5, 2.0))
    go("motor frictionloss DR", dr_friction_loss=True)
    go("dt jitter ±5%", dr_dt_jitter=0.05)

    print("\n--- combined ---")
    go("delay [1,3] only", dr_delay=(1, 3))
    go("delay [1,3] + accel DR", dr_delay=(1, 3), dr_accel=DR_MOTOR_ACCEL_RANGE_RAD_S2)


if __name__ == "__main__":
    main()
