"""Flash LowLevelServer.ino onto the Nano only if it isn't already running
byte-for-byte the same sketch — skips the ~15-30s compile+upload cycle on
every normal demo boot where the Nano already has the right firmware.

Compares a hash the firmware reports over CMD_GET_FIRMWARE_VERSION (baked
in at compile time by LowLevelServer/gen_firmware_version.py) against a
fresh hash computed from the local .ino/.h source — see that script for
what's hashed.

Usage:
    python flash_if_needed.py [--port COM3] [--vid 1A86] [--pid 7523] [--fqbn ...]

If --port is omitted, the Nano is auto-discovered by USB VID:PID (see
pi_demo_common.py) — works the same way on Linux, macOS, and Windows.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pi_demo_common import DEFAULT_PID, DEFAULT_VID, REPO_ROOT, find_nano_port

from lowlevel_client import LowLevelClient  # noqa: E402  (path set up by pi_demo_common)

SKETCH_DIR = REPO_ROOT / "RotaryInvertedPendulum-arduino" / "LowLevelServer"
GEN_SCRIPT = SKETCH_DIR / "gen_firmware_version.py"
VERSION_HEADER = SKETCH_DIR / "firmware_version.h"

DEFAULT_FQBN = "arduino:avr:nano:cpu=atmega328"


def _local_expected_hash() -> int:
    # Regenerate rather than just reading the checked-in header, so a
    # local edit to the sketch that forgot to re-run the generator still
    # gets flashed instead of silently "matching" a stale hash.
    subprocess.run([sys.executable, str(GEN_SCRIPT)], check=True, cwd=SKETCH_DIR)
    for line in VERSION_HEADER.read_text().splitlines():
        if line.startswith("#define FIRMWARE_VERSION_HASH"):
            return int(line.split()[2].rstrip("UL"), 16)
    raise RuntimeError(f"{VERSION_HEADER} missing FIRMWARE_VERSION_HASH")


def _device_hash(port: str) -> int | None:
    with LowLevelClient(port) as client:
        if not client.wait_until_ready():
            return None
        return client.get_firmware_version()


def _flash(port: str, fqbn: str) -> None:
    print("[pi_demo] Flashing LowLevelServer.ino...")
    subprocess.run(
        ["arduino-cli", "compile", "--upload", "-p", port, "--fqbn", fqbn,
         str(SKETCH_DIR)],
        check=True,
    )


def ensure_flashed(port: str | None = None, fqbn: str = DEFAULT_FQBN,
                    vid: int = DEFAULT_VID, pid: int = DEFAULT_PID) -> str:
    """Flash if needed; returns the port actually used (handy when the
    caller passed None and this had to auto-discover it)."""
    if port is None:
        port = find_nano_port(vid, pid)

    expected = _local_expected_hash()
    actual = _device_hash(port)

    if actual == expected:
        print(f"[pi_demo] Firmware up to date (0x{expected:08X}), skipping flash.")
        return port

    if actual is None:
        print("[pi_demo] Nano didn't answer READY (blank board or wrong sketch) — flashing.")
    else:
        print(f"[pi_demo] Firmware mismatch (have 0x{actual:08X}, want 0x{expected:08X}) — flashing.")

    _flash(port, fqbn)

    # Re-check after the flash's own reset so a silently failed upload is
    # caught here instead of surfacing later as a run_policy.py protocol error.
    actual = _device_hash(port)
    if actual != expected:
        got = "no response" if actual is None else f"0x{actual:08X}"
        raise RuntimeError(
            f"Flash verification failed: device reports {got}, expected 0x{expected:08X}"
        )
    print("[pi_demo] Flash verified.")
    return port


def _parse_hex(s: str) -> int:
    return int(s, 16)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=None,
                   help="serial port; auto-discovered via VID:PID if omitted")
    p.add_argument("--vid", type=_parse_hex, default=DEFAULT_VID, help="USB vendor ID, hex")
    p.add_argument("--pid", type=_parse_hex, default=DEFAULT_PID, help="USB product ID, hex")
    p.add_argument("--fqbn", default=DEFAULT_FQBN)
    args = p.parse_args(argv)
    ensure_flashed(args.port, args.fqbn, args.vid, args.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
