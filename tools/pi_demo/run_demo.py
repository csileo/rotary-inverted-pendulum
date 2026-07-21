"""Auto-demo launcher: tolerates whatever order and delay the person plugs
in the Pi/computer's power, the pendulum's 12V adapter, and the Nano's USB
cable in. Waits for the Nano to show up (by USB VID:PID, not a fixed device
path, so this runs unchanged on Linux, macOS, and Windows — no udev rule,
no COM-port-vs-/dev-path branching), flashes it only if its firmware
doesn't already match, waits for 12V motor power to actually be present,
then runs the reference policy. See README.md in this directory.

Usage:
    python run_demo.py

Configuration is via environment variables (all optional — see README.md).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from check_motor_power import motor_power_present
from flash_if_needed import DEFAULT_FQBN, ensure_flashed
from pi_demo_common import DEFAULT_PID, DEFAULT_VID, RL_DIR, find_nano_port


def main() -> int:
    port = os.environ.get("PENDULUM_PORT")
    vid = int(os.environ.get("PENDULUM_USB_VID", f"{DEFAULT_VID:X}"), 16)
    pid = int(os.environ.get("PENDULUM_USB_PID", f"{DEFAULT_PID:X}"), 16)
    policy = os.environ.get(
        "PENDULUM_POLICY", str(RL_DIR / "models" / "policy_working_balance.zip"))
    frame_stack = os.environ.get("PENDULUM_FRAME_STACK", "3")
    duration_s = os.environ.get("PENDULUM_DURATION_S", "60")
    motor_power_timeout_s = float(os.environ.get("PENDULUM_MOTOR_POWER_TIMEOUT_S", "120"))

    if port is None:
        print("[run_demo] Waiting for the Nano...")
        port = find_nano_port(vid, pid)
    print(f"[run_demo] Using port {port}")

    print("[run_demo] Checking / flashing firmware...")
    ensure_flashed(port, DEFAULT_FQBN, vid, pid)

    print("[run_demo] Waiting for 12V motor power...")
    deadline = time.monotonic() + motor_power_timeout_s
    while not motor_power_present(port, vid, pid):
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


if __name__ == "__main__":
    raise SystemExit(main())
