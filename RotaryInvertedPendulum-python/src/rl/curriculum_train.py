"""
Curriculum trainer -- Windows-compatible replacement for curriculum_train.sh.

Usage:
    python curriculum_train.py [prefix]          # prefix defaults to "curriculum"
    python curriculum_train.py curriculum4

Env-var overrides:
    STEPS_PER_STAGE   (default 200000)
    STEPS_LAST_STAGE  (default = STEPS_PER_STAGE) steps for the last DR stage only
    N_ENVS            (default 16)
    GRADIENT_STEPS    (default -1)
    CONTROL_FREQ      (default 50)
    DEVICE            (default cuda)
    INITIAL_RESUME    if set, skip stage 1 and use this model as starting point
    LAG_RANGES        comma-separated "min:max" pairs in seconds for DR stages
                      default "0.000:0.030,0.010:0.030"  (original 2-stage schedule)
"""
import os
import subprocess
import sys


def env_int(name, default):
    return int(os.environ.get(name, default))

def env_str(name, default):
    return os.environ.get(name, default)


PREFIX         = sys.argv[1] if len(sys.argv) > 1 else "curriculum"
STEPS          = env_int("STEPS_PER_STAGE", 200000)
STEPS_LAST     = env_int("STEPS_LAST_STAGE", STEPS)
N_ENVS         = env_int("N_ENVS", 16)
GRAD_STEPS     = env_int("GRADIENT_STEPS", -1)
CONTROL_FREQ   = env_int("CONTROL_FREQ", 50)
DEVICE         = env_str("DEVICE", "auto")
INITIAL_RESUME   = env_str("INITIAL_RESUME", None)
LAG_RANGES_STR   = env_str("LAG_RANGES", "0.000:0.030,0.010:0.030")
MOTOR_JERK_WEIGHT = env_str("MOTOR_JERK_WEIGHT", None)
FRAME_STACK      = env_int("FRAME_STACK", 3)

lag_ranges = [tuple(map(float, r.split(":"))) for r in LAG_RANGES_STR.split(",")]

BASE = [
    sys.executable, "-u", "train_sac.py",
    "--control-freq", str(CONTROL_FREQ),
    "--n-envs", str(N_ENVS),
    "--device", DEVICE,
    "--frame-stack", str(FRAME_STACK),
]
# --gradient-steps n'est utile que pour n_envs > 1 (pour n_envs=1, train_sac.py
# l'ignore et utilise toujours 1 — identique à origin/main)
if N_ENVS > 1:
    BASE += ["--gradient-steps", str(GRAD_STEPS)]
if MOTOR_JERK_WEIGHT is not None:
    BASE += ["--reward-motor-jerk-weight", MOTOR_JERK_WEIGHT]

stages = []

if INITIAL_RESUME:
    prev_model = INITIAL_RESUME
    stage_num = 2
else:
    S1 = f"{PREFIX}_stage1"
    stages.append(("Stage 1 -- no DR", S1,
                   BASE + ["--total-steps", str(STEPS), "--run-name", S1]))
    prev_model = f"runs/{S1}/best_model.zip"
    stage_num = 2

for i, (lag_min, lag_max) in enumerate(lag_ranges):
    is_last = (i == len(lag_ranges) - 1)
    steps = STEPS_LAST if is_last else STEPS
    Sn = f"{PREFIX}_stage{stage_num}"
    label = (f"Stage {stage_num} -- "
             f"action-lag [{lag_min*1000:.0f}-{lag_max*1000:.0f} ms]"
             + (" (last)" if is_last and STEPS_LAST != STEPS else ""))
    cmd = BASE + [
        "--total-steps", str(steps),
        "--run-name", Sn,
        "--domain-randomization",
        "--dr-action-lag-tau-min", str(lag_min),
        "--dr-action-lag-tau-max", str(lag_max),
        "--resume", prev_model,
    ]
    stages.append((label, Sn, cmd))
    prev_model = f"runs/{Sn}/best_model.zip"
    stage_num += 1

last_stage = f"{PREFIX}_stage{stage_num - 1}"

steps_info = f"steps/stage={STEPS}" + (f" last={STEPS_LAST}" if STEPS_LAST != STEPS else "")
print(f"Curriculum: prefix={PREFIX}, {steps_info}, n_envs={N_ENVS}, "
      f"grad_steps={GRAD_STEPS}, freq={CONTROL_FREQ} Hz, device={DEVICE}")
if INITIAL_RESUME:
    print(f"Stage 1 skipped -- starting from {INITIAL_RESUME}")
print()

for label, _, cmd in stages:
    print(f"=== {label} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStage failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print()

print("=== Curriculum complete ===")
print(f"Deploy: python run_policy.py --policy runs/{last_stage}/best_model.zip "
      f"--port <PORT> --control-freq {CONTROL_FREQ} --log logs/run_{PREFIX}.npz")
