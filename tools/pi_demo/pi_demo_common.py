"""Shared helpers for the auto-demo launcher (run_demo.py).

flash_if_needed.py, check_motor_power.py, and run_demo.py all need to find
the Nano's serial device and talk LowLevelServer's protocol. Device
discovery goes through pyserial's serial.tools.list_ports, which is
cross-platform (Linux, macOS, Windows) — there's no OS-specific setup step
(no udev rule, no fixed /dev/ttyUSB0-vs-COM3 assumption to maintain), so
the same code runs unchanged on a headless Pi or a Windows laptop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from serial.tools import list_ports

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_DIR = REPO_ROOT / "RotaryInvertedPendulum-python" / "src" / "rl"
sys.path.insert(0, str(RL_DIR))

# CH340/CH341 — the USB-serial chip on most Nano clones. Override via the
# --vid/--pid flags (or PENDULUM_USB_VID/PENDULUM_USB_PID env vars in
# run_demo.py) if your board uses a different one, e.g.:
#   CP2102:     10C4:EA60
#   FTDI FT232: 0403:6001
# Find yours on any OS with: python -m serial.tools.list_ports -v
DEFAULT_VID = 0x1A86
DEFAULT_PID = 0x7523


def find_nano_port(vid: int = DEFAULT_VID, pid: int = DEFAULT_PID,
                    timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> str:
    """Poll for a USB-serial device matching (vid, pid); return its OS port
    name (e.g. /dev/ttyUSB0, /dev/cu.usbserial-*, or COM3).

    Waits rather than failing on the first miss — the person setting up
    the demo may plug in the USB cable well after starting this script, in
    any order relative to the pendulum's 12V supply.
    """
    deadline = time.monotonic() + timeout_s
    printed = False
    while time.monotonic() < deadline:
        matches = [p.device for p in list_ports.comports()
                   if p.vid == vid and p.pid == pid]
        if matches:
            return matches[0]
        if not printed:
            print(f"[pi_demo] Waiting for a {vid:04X}:{pid:04X} USB-serial "
                  "device (plug in the Nano's USB cable)...")
            printed = True
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"Timed out waiting for a {vid:04X}:{pid:04X} USB-serial device. "
        "Is the Nano plugged in? `python -m serial.tools.list_ports -v` "
        "lists what's actually attached and its VID:PID."
    )
