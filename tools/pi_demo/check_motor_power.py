"""Check that the pendulum's 12V motor power is actually present, using the
AS5600 encoder already on the rig instead of new sensing hardware.

There's no voltage sensing on the 12V rail (see docs/electronics_design.md),
so this checks it indirectly: issue a small, brief acceleration pulse and
confirm the encoder actually saw the motor move. Without 12V the DRV8825
still receives STEP/ENABLE pulses from the Nano (which runs off USB power
independently of the motor rail), but the coils have no current, so the
shaft doesn't turn and the encoder delta stays near the noise floor.

Usage:
    python check_motor_power.py [--port COM3] [--vid 1A86] [--pid 7523]
Exit code 0 if power was detected, 1 otherwise (for a retry loop).

If --port is omitted, the Nano is auto-discovered by USB VID:PID (see
pi_demo_common.py) — works the same way on Linux, macOS, and Windows.
"""

from __future__ import annotations

import argparse
import time

from pi_demo_common import DEFAULT_PID, DEFAULT_VID, find_nano_port

from lowlevel_client import LowLevelClient  # noqa: E402  (path set up by pi_demo_common)

# A powered motor clears this easily; a stationary one (no 12V) stays
# within encoder quantisation + bearing play (AS5600 is ~0.088°/count,
# see docs/electronics_design.md) with wide margin either way.
MOVEMENT_THRESHOLD_RAD = 0.03
TEST_ACCEL_RAD_S2 = 20.0  # gentle — this is a presence check, not a swing
TEST_DURATION_S = 0.1


def motor_power_present(port: str | None = None, vid: int = DEFAULT_VID,
                         pid: int = DEFAULT_PID) -> bool:
    if port is None:
        port = find_nano_port(vid, pid)

    with LowLevelClient(port) as client:
        if not client.wait_until_ready():
            raise RuntimeError("Nano did not respond to READY")

        client.set_acceleration(0.0)
        client.engage_motor()
        try:
            start = client.get_state()
            client.set_acceleration(TEST_ACCEL_RAD_S2)
            time.sleep(TEST_DURATION_S)
            end = client.get_state()
        finally:
            # disengage_motor() forceStop()s the stepper immediately
            # regardless of residual commanded velocity — see
            # LowLevelServer.ino's CMD_DISENGAGE_MOTOR handler.
            client.set_acceleration(0.0)
            client.disengage_motor()

    moved = abs(end.motor_pos_rad - start.motor_pos_rad)
    return moved >= MOVEMENT_THRESHOLD_RAD


def _parse_hex(s: str) -> int:
    return int(s, 16)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=None,
                   help="serial port; auto-discovered via VID:PID if omitted")
    p.add_argument("--vid", type=_parse_hex, default=DEFAULT_VID, help="USB vendor ID, hex")
    p.add_argument("--pid", type=_parse_hex, default=DEFAULT_PID, help="USB product ID, hex")
    args = p.parse_args(argv)
    ok = motor_power_present(args.port, args.vid, args.pid)
    print("[pi_demo] Motor power: PRESENT" if ok else "[pi_demo] Motor power: NOT DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
