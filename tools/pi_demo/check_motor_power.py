"""Check that the pendulum's 12V motor power is actually present, using the
AS5600 encoder already on the rig instead of new sensing hardware.

There's no voltage sensing on the 12V rail (see docs/electronics_design.md),
so this checks it indirectly: issue a small, brief acceleration pulse and
confirm the encoder actually saw the motor move. Without 12V the DRV8825
still receives STEP/ENABLE pulses from the Nano (which runs off USB power
independently of the motor rail), but the coils have no current, so the
shaft doesn't turn and the encoder delta stays near the noise floor.

Usage:
    python check_motor_power.py [--port COM3]
Exit code 0 if power was detected, 1 otherwise (for a retry loop).

If --port is omitted, the Nano is auto-discovered by USB vid/pid (see
pi_demo_common.py) — works the same way on Linux, macOS, and Windows.
"""

from __future__ import annotations

import argparse
import time

from pi_demo_common import find_nano_port

from lowlevel_client import LowLevelClient  # noqa: E402  (path set up by pi_demo_common)

# A powered motor clears this easily; a stationary one (no 12V) stays
# within encoder quantisation + bearing play (AS5600 is ~0.088°/count,
# see docs/electronics_design.md) with wide margin either way.
MOVEMENT_THRESHOLD_RAD = 0.03
TEST_ACCEL_RAD_S2 = 20.0  # gentle — this is a presence check, not a swing
TEST_DURATION_S = 0.1


def _pulse_and_check(client: LowLevelClient) -> bool:
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


def motor_power_present(port: str | None = None, *,
                         client: LowLevelClient | None = None) -> bool:
    """Check for 12V motor power via a brief acceleration pulse.

    Pass an already-open `client` (e.g. from `tools/pi_demo/run_demo.py`'s
    persistent connection) to avoid opening a fresh serial connection —
    each new connection resets the Arduino (~2s), which is wasteful when
    this is polled repeatedly while waiting for the 12V adapter to be
    plugged in. Falls back to opening its own connection (and closing it
    afterward) for standalone CLI use.
    """
    if client is not None:
        return _pulse_and_check(client)

    if port is None:
        port = find_nano_port()
    with LowLevelClient(port) as client:
        return _pulse_and_check(client)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=None,
                   help="serial port; auto-discovered via usb_config.json if omitted")
    args = p.parse_args(argv)
    ok = motor_power_present(args.port)
    print("[pi_demo] Motor power: PRESENT" if ok else "[pi_demo] Motor power: NOT DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
