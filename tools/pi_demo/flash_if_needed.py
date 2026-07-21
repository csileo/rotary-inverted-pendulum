"""Flash LowLevelServer.ino onto the Nano only if it isn't already running
byte-for-byte the same sketch — skips the ~15-30s compile+upload cycle on
every normal demo boot where the Nano already has the right firmware.

Compares a hash the firmware reports over CMD_GET_FIRMWARE_VERSION (baked
in at compile time by LowLevelServer/gen_firmware_version.py) against a
fresh hash computed from the local .ino/.h source — see that script for
what's hashed.

Also requires RotaryInvertedPendulum-arduino/LowLevelServer/hw_config.h to
exist before compiling: it selects which AS5600 I2C backend to build (see
hw_profiles/ in that directory) and there is no safe default to guess —
building with the wrong one for this rig's encoder module can corrupt
sensor readings feeding a live motor control loop.

Usage:
    python flash_if_needed.py [--port COM3] [--fqbn ...]

If --port is omitted, the Nano is auto-discovered by USB vid/pid (see
pi_demo_common.py) — works the same way on Linux, macOS, and Windows.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pi_demo_common import REPO_ROOT, find_nano_port

from lowlevel_client import LowLevelClient  # noqa: E402  (path set up by pi_demo_common)

SKETCH_DIR = REPO_ROOT / "RotaryInvertedPendulum-arduino" / "LowLevelServer"
GEN_SCRIPT = SKETCH_DIR / "gen_firmware_version.py"
VERSION_HEADER = SKETCH_DIR / "firmware_version.h"
HW_CONFIG_PATH = SKETCH_DIR / "hw_config.h"
HW_PROFILES_DIR = SKETCH_DIR / "hw_profiles"

DEFAULT_FQBN = "arduino:avr:nano:cpu=atmega328"


def _ensure_hw_config() -> None:
    if HW_CONFIG_PATH.exists():
        return
    available = sorted(p.name for p in HW_PROFILES_DIR.glob("*.h")) if HW_PROFILES_DIR.exists() else []
    raise RuntimeError(
        f"{HW_CONFIG_PATH} is missing. This selects which AS5600 I2C "
        "backend to compile — there is no safe default (the wrong one for "
        "this rig's encoder module can corrupt sensor readings). Copy one "
        f"of {available or '<no profiles found>'} from {HW_PROFILES_DIR} "
        "to hw_config.h, matching your AS5600 module (see docs/BOM.md)."
    )


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


def ensure_flashed(port: str | None = None, fqbn: str = DEFAULT_FQBN) -> str:
    """Flash if needed; returns the port actually used (handy when the
    caller passed None and this had to auto-discover it)."""
    _ensure_hw_config()

    if port is None:
        port = find_nano_port()

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=None,
                   help="serial port; auto-discovered via usb_config.json if omitted")
    p.add_argument("--fqbn", default=DEFAULT_FQBN)
    args = p.parse_args(argv)
    ensure_flashed(args.port, args.fqbn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
