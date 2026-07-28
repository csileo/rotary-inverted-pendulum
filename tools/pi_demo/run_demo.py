"""Auto-demo launcher: tolerates whatever order and delay the person plugs
in the Pi/computer's power, the pendulum's 12V adapter, and the Nano's USB
cable in. Waits for the Nano to show up (by USB vid/pid, not a fixed device
path, so this runs unchanged on Linux, macOS, and Windows — no udev rule,
no COM-port-vs-/dev-path branching), flashes it only if its firmware
doesn't already match, waits for 12V motor power to actually be present,
then runs the reference policy — on repeat, forever, so a headless/no-network
Pi keeps re-demoing without anyone reconnecting to restart it by hand. See
README.md in this directory.

Usage:
    python run_demo.py

Configuration is via environment variables (all optional — see README.md)
for the policy/duration/loop-delay to run. Which Nano to talk to is never
one of them — that only ever comes from usb_config.json (see
pi_demo_common.py). Ctrl-C (or SIGTERM, e.g. `systemctl stop`) is the only
way to actually stop the loop.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from check_motor_power import motor_power_present
from flash_if_needed import DEFAULT_FQBN, ensure_flashed
from pi_demo_common import RL_DIR, find_nano_port


def run_once(policy: str, frame_stack: str, duration_s: str,
             motor_power_timeout_s: float) -> int:
    print("[run_demo] Waiting for the Nano...")
    port = find_nano_port()
    print(f"[run_demo] Using port {port}")

    print("[run_demo] Checking / flashing firmware...")
    ensure_flashed(port, DEFAULT_FQBN)

    print("[run_demo] Waiting for 12V motor power...")
    deadline = time.monotonic() + motor_power_timeout_s
    while not motor_power_present(port):
        if time.monotonic() >= deadline:
            print("[run_demo] Timed out waiting for motor power. "
                  "Is the 12V adapter plugged in?", file=sys.stderr)
            return 1
        print("[run_demo] No motor movement detected — is the 12V adapter "
              "plugged in? Retrying in 3s...")
        time.sleep(3)

    print("[run_demo] Motor power confirmed. Starting policy.")
    result = subprocess.run(
        [sys.executable, "run_policy.py", "--policy", policy, "--port", port,
         "--frame-stack", frame_stack, "--duration-s", duration_s],
        cwd=RL_DIR,
    )
    return result.returncode


def main() -> int:
    policy = os.environ.get(
        "PENDULUM_POLICY",
        str(RL_DIR / "models" / "distill_working_balance_h32_dagger" / "student.pt"))
    frame_stack = os.environ.get("PENDULUM_FRAME_STACK", "3")
    duration_s = os.environ.get("PENDULUM_DURATION_S", "60")
    motor_power_timeout_s = float(os.environ.get("PENDULUM_MOTOR_POWER_TIMEOUT_S", "120"))
    loop_delay_s = float(os.environ.get("PENDULUM_LOOP_DELAY_S", "10"))

    try:
        while True:
            try:
                code = run_once(policy, frame_stack, duration_s, motor_power_timeout_s)
                status = "ok" if code == 0 else f"exit code {code}"
            except Exception as exc:  # keep looping no matter what fails
                status = f"error: {exc}"
            print(f"[run_demo] Cycle finished ({status}). "
                  f"Restarting in {loop_delay_s:.0f}s — Ctrl-C to stop.")
            time.sleep(loop_delay_s)
    except KeyboardInterrupt:
        print("[run_demo] Interrupted — stopping.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
