"""Shared helpers for the auto-demo launcher (run_demo.py).

flash_if_needed.py, check_motor_power.py, and run_demo.py all need to find
the Nano's serial device and talk LowLevelServer's protocol. Device
discovery goes through pyserial's serial.tools.list_ports, which is
cross-platform (Linux, macOS, Windows) — there's no OS-specific setup step
(no udev rule, no fixed /dev/ttyUSB0-vs-COM3 assumption to maintain), so
the same code runs unchanged on a headless Pi or a Windows laptop.

Which vid/pid to look for comes from usb_config.json if present (this
machine's own detected chip, written by detect_usb_config.py), else falls
back to usb_profiles/ch340.json (the reference default — see that file).
The only way to change which chip is used is to overwrite usb_config.json
(re-run detect_usb_config.py, or edit it by hand) — there is no env var or
CLI override, so nothing can silently drift out of sync with what a given
rig actually reports.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from serial.tools import list_ports

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_DIR = REPO_ROOT / "RotaryInvertedPendulum-python" / "src" / "rl"
sys.path.insert(0, str(RL_DIR))

PI_DEMO_DIR = Path(__file__).resolve().parent
USB_CONFIG_PATH = PI_DEMO_DIR / "usb_config.json"
USB_DEFAULT_PROFILE_PATH = PI_DEMO_DIR / "usb_profiles" / "ch340.json"


def _parse_hex(value: str) -> int:
    return int(value, 16)


def load_usb_ids() -> tuple[int, int]:
    """Return (vid, pid) to search for.

    usb_config.json (this machine's own detected chip) wins if present;
    otherwise falls back to the tracked default profile.
    """
    path = USB_CONFIG_PATH if USB_CONFIG_PATH.exists() else USB_DEFAULT_PROFILE_PATH
    with open(path) as f:
        doc = json.load(f)
    return _parse_hex(doc["vid"]), _parse_hex(doc["pid"]), path


def find_nano_port(timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> str:
    """Poll for a USB-serial device matching the configured (vid, pid);
    return its OS port name (e.g. /dev/ttyUSB0, /dev/cu.usbserial-*, or COM3).

    Waits rather than failing on the first miss — the person setting up
    the demo may plug in the USB cable well after starting this script, in
    any order relative to the pendulum's 12V supply.
    """
    vid, pid, source = load_usb_ids()
    deadline = time.monotonic() + timeout_s
    printed = False
    while time.monotonic() < deadline:
        matches = [p.device for p in list_ports.comports()
                   if p.vid == vid and p.pid == pid]
        if matches:
            return matches[0]
        if not printed:
            print(f"[pi_demo] Waiting for a {vid:04X}:{pid:04X} USB-serial "
                  f"device (source: {source.relative_to(REPO_ROOT)}) — "
                  "plug in the Nano's USB cable...")
            printed = True
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"Timed out waiting for a {vid:04X}:{pid:04X} USB-serial device "
        f"(source: {source.relative_to(REPO_ROOT)}). Is the Nano plugged "
        "in? If you've changed hardware, run detect_usb_config.py to "
        "refresh usb_config.json, or check tools/pi_demo/usb_profiles/ for "
        "other known chips. `python -m serial.tools.list_ports -v` lists "
        "what's actually attached and its vid:pid."
    )
