"""Turn an on-policy real-rig rollout log into fresh (obs, action_target) data.

DAgger-style covariate-shift fix for distillation: `distill.py`'s dataset is
built from the *teacher's own* real-rig replay buffer (states the teacher
visited while balancing well), so the student never sees states specific to
*its own* mistakes (e.g. the recovery window right after a near-fall). This
script closes that gap:

    1. Load a trajectory log produced by `run_policy.py --log ...` while
       running the STUDENT (or QAT student) closed-loop on the real rig —
       i.e. exactly the states the deployed policy actually visits,
       including its failure modes.
    2. Reconstruct the frame-stacked observation at each tick (same
       `FrameStacker` class `run_policy.py` used live, so this is a
       bit-exact replay of what the policy actually saw — not an
       approximation).
    3. Relabel every one of those observations with the TEACHER's
       deterministic action (not the student's own logged action) — the
       actual distillation target signal.
    4. Optionally concatenate with an existing dataset.npz (e.g. the
       original teacher-replay-buffer dataset) and save the merged result,
       ready for `distill.py`'s train stage (drop it in as
       `<out-dir>/dataset.npz` and `distill.py` will skip straight to
       training since the dataset already exists).

Usage:
    python dagger_relabel.py \\
        --teacher models/policy_working_balance.zip \\
        --rollout-log logs/int8_closed_loop.npz \\
        --frame-stack 3 \\
        --base-dataset models/distill_working_balance_h32/dataset.npz \\
        --out models/distill_working_balance_h32_dagger/dataset.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from frame_stack import FrameStacker


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def build_raw_obs(log: dict) -> np.ndarray:
    """Reconstruct the 6-dim raw observation at every logged tick.

    Matches `pendulum_env.py:_obs()` / `run_policy.py:make_obs()`:
    [motor_pos, sin(theta), cos(theta), motor_vel, pen_vel, prev_action].
    `prev_action[t] = action[t-1]` (0.0 at t=0), matching the live loop's
    ordering — the action logged at tick t was computed *from* raw_obs[t],
    which itself used the action from tick t-1.
    """
    motor_pos = np.asarray(log["motor_pos_rad"], dtype=np.float32)
    phi = np.asarray(log["pendulum_pos_rad"], dtype=np.float32)
    motor_vel = np.asarray(log["motor_vel_rad_s"], dtype=np.float32)
    pen_vel = np.asarray(log["pendulum_vel_rad_s"], dtype=np.float32)
    action = np.asarray(log["action"], dtype=np.float32)

    prev_action = np.empty_like(action)
    prev_action[0] = 0.0
    prev_action[1:] = action[:-1]

    theta = _wrap_pi(phi - math.pi)
    raw_obs = np.stack(
        [motor_pos, np.sin(theta), np.cos(theta), motor_vel, pen_vel, prev_action],
        axis=1,
    ).astype(np.float32)
    return raw_obs


def stack_frames(raw_obs: np.ndarray, frame_stack: int) -> np.ndarray:
    """Bit-exact replay of the live FrameStacker: reset() on the first
    frame, push() on every subsequent one."""
    frame_dim = raw_obs.shape[1]
    stacker = FrameStacker(frame_stack, frame_dim=frame_dim)
    out = np.empty((raw_obs.shape[0], frame_dim * frame_stack), dtype=np.float32)
    out[0] = stacker.reset(raw_obs[0])
    for t in range(1, raw_obs.shape[0]):
        out[t] = stacker.push(raw_obs[t])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Relabel a real-rig on-policy rollout with the teacher's "
                    "actions, for a DAgger-style distillation refresh"
    )
    p.add_argument("--teacher", required=True, type=Path,
                   help="path to the SAC teacher .zip (relabelling source)")
    p.add_argument("--rollout-log", required=True, type=Path,
                   help="trajectory .npz produced by `run_policy.py --log ...` "
                        "while running the student closed-loop on the rig")
    p.add_argument("--frame-stack", type=int, default=3,
                   help="MUST match the frame stack the rollout policy (and "
                        "the teacher) were built with")
    p.add_argument("--base-dataset", type=Path, default=None,
                   help="existing dataset.npz (obs, action_target) to "
                        "concatenate the new data onto; omit for new-data-only")
    p.add_argument("--out", required=True, type=Path,
                   help="output dataset.npz path")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    print(f"[dagger] loading rollout log: {args.rollout_log}")
    log = dict(np.load(args.rollout_log))
    n = log["action"].shape[0]
    print(f"[dagger] {n} logged ticks "
          f"(control_freq_hz={float(log['control_freq_hz']):.1f}, "
          f"policy={log['policy_path']})")

    raw_obs = build_raw_obs(log)
    stacked_obs = stack_frames(raw_obs, args.frame_stack)
    print(f"[dagger] reconstructed obs: {stacked_obs.shape} "
          f"(frame_stack={args.frame_stack})")

    print(f"[dagger] loading teacher: {args.teacher}")
    teacher = SAC.load(str(args.teacher), device=args.device)
    expected_dim = teacher.observation_space.shape[0]
    if stacked_obs.shape[1] != expected_dim:
        raise RuntimeError(
            f"reconstructed obs is {stacked_obs.shape[1]}-dim but the teacher "
            f"expects {expected_dim}-dim — wrong --frame-stack?"
        )

    action_target, _ = teacher.predict(stacked_obs, deterministic=True)
    action_target = np.asarray(action_target, dtype=np.float32).reshape(n, -1)
    print(f"[dagger] relabelled {n} observations with the teacher's "
          f"deterministic action")

    if args.base_dataset is not None:
        print(f"[dagger] merging with base dataset: {args.base_dataset}")
        base = np.load(args.base_dataset)
        base_obs = np.asarray(base["obs"], dtype=np.float32)
        base_act = np.asarray(base["action_target"], dtype=np.float32)
        if base_obs.shape[1] != stacked_obs.shape[1]:
            raise RuntimeError(
                f"base dataset obs is {base_obs.shape[1]}-dim, new data is "
                f"{stacked_obs.shape[1]}-dim — mismatched frame-stack/obs contract"
            )
        obs_out = np.concatenate([base_obs, stacked_obs], axis=0)
        act_out = np.concatenate([base_act, action_target], axis=0)
        print(f"[dagger] merged dataset: {obs_out.shape[0]} samples "
              f"({base_obs.shape[0]} base + {stacked_obs.shape[0]} new on-policy)")
    else:
        obs_out, act_out = stacked_obs, action_target

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=obs_out, action_target=act_out)
    print(f"[dagger] saved -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
