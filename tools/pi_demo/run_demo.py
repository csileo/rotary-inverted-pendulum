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

Performance note: earlier versions of this script re-flashed-checked,
re-opened a fresh serial connection, and re-ran `run_policy.py` as a brand
new subprocess for every single cycle. Each of those cost real time: every
new serial connection resets the Arduino (~2s: bootloader + `setup()`), and
`run_policy.py` importing torch/stable-baselines3 from scratch is slow on a
Pi 3B+. This version instead: loads the policy ONCE, keeps ONE
LowLevelClient connection open across every cycle, and calls
`run_policy.py`'s `run_control_loop()` directly in-process instead of
shelling out. The Nano is only rediscovered / reflashed / reconnected if
something actually goes wrong (USB unplug, a serial error, etc.) — see
`main()`'s outer loop. Pointing `PENDULUM_POLICY` at the `.npz` (not `.pt`
or `.zip`) checkpoint also means this process never imports torch at all —
see `numpy_student.py` / `export_weights_numpy.py`.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from check_motor_power import motor_power_present
from flash_if_needed import DEFAULT_FQBN, ensure_flashed
from pi_demo_common import RL_DIR, find_nano_port

from lowlevel_client import LowLevelClient  # noqa: E402  (path set up by pi_demo_common)
from run_policy import load_policy, run_control_loop  # noqa: E402

# Set by the SIGINT/SIGTERM handler; every wait/sleep in this file checks it
# instead of a plain time.sleep(), so Ctrl-C / `systemctl stop` interrupts
# immediately regardless of which phase (waiting for the Nano, waiting for
# 12V, mid-balance, or paused between cycles) the loop is currently in.
_stop_event = threading.Event()


def _handle_stop_signal(*_):
    _stop_event.set()


def run_once(client: LowLevelClient, model, *, frame_stack: int,
             control_freq: float, max_accel_rad_s2: float, duration_s: str,
             motor_power_timeout_s: float, policy_label: str) -> int:
    print("[run_demo] Waiting for 12V motor power...")
    deadline = time.monotonic() + motor_power_timeout_s
    while not motor_power_present(client=client):
        if _stop_event.is_set():
            return 0
        if time.monotonic() >= deadline:
            print("[run_demo] Timed out waiting for motor power. "
                  "Is the 12V adapter plugged in?", file=sys.stderr)
            return 1
        print("[run_demo] No motor movement detected — is the 12V adapter "
              "plugged in? Retrying in 3s...")
        _stop_event.wait(3)

    if _stop_event.is_set():
        return 0

    print("[run_demo] Motor power confirmed. Starting policy.")
    return run_control_loop(
        model, client,
        frame_stack=frame_stack,
        control_freq=control_freq,
        max_accel_rad_s2=max_accel_rad_s2,
        duration_s=float(duration_s),
        policy_label=policy_label,
        stop_event=_stop_event,
    )


def main() -> int:
    policy = os.environ.get(
        "PENDULUM_POLICY",
        str(RL_DIR / "models" / "distill_working_balance_h32_dagger" / "student_numpy.npz"))
    frame_stack = int(os.environ.get("PENDULUM_FRAME_STACK", "3"))
    duration_s = os.environ.get("PENDULUM_DURATION_S", "60")
    motor_power_timeout_s = float(os.environ.get("PENDULUM_MOTOR_POWER_TIMEOUT_S", "120"))
    loop_delay_s = float(os.environ.get("PENDULUM_LOOP_DELAY_S", "10"))
    control_freq = float(os.environ.get("PENDULUM_CONTROL_FREQ_HZ", "35.0"))
    max_accel_rad_s2 = float(os.environ.get("PENDULUM_MAX_ACCEL_RAD_S2", "150.0"))

    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    print(f"[run_demo] Loading policy: {policy}")
    model = load_policy(policy)

    while not _stop_event.is_set():
        try:
            print("[run_demo] Waiting for the Nano...")
            port = find_nano_port()
            print(f"[run_demo] Using port {port}")

            print("[run_demo] Checking / flashing firmware...")
            ensure_flashed(port, DEFAULT_FQBN)

            # One connection, reused for every cycle below — only torn down
            # (by falling out of this `with` block) if something actually
            # raises, e.g. the Nano gets unplugged mid-demo.
            with LowLevelClient(port) as client:
                while not _stop_event.is_set():
                    code = run_once(
                        client, model,
                        frame_stack=frame_stack, control_freq=control_freq,
                        max_accel_rad_s2=max_accel_rad_s2, duration_s=duration_s,
                        motor_power_timeout_s=motor_power_timeout_s,
                        policy_label=policy,
                    )
                    if _stop_event.is_set():
                        break
                    status = "ok" if code == 0 else f"exit code {code}"
                    print(f"[run_demo] Cycle finished ({status}). "
                          f"Restarting in {loop_delay_s:.0f}s — Ctrl-C to stop.")
                    _stop_event.wait(loop_delay_s)
        except Exception as exc:
            if _stop_event.is_set():
                break
            print(f"[run_demo] Lost connection or setup failed ({exc}). "
                  f"Retrying in {loop_delay_s:.0f}s.", file=sys.stderr)
            _stop_event.wait(loop_delay_s)

    print("[run_demo] Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
