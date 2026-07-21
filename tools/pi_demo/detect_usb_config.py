"""Detect this machine's connected Nano and write tools/pi_demo/usb_config.json.

Run this once per machine/rig (or again after swapping to a Nano with a
different USB-serial chip). Assumes exactly one relevant USB-serial device
is connected — see the project's demo/dev usage: a single Nano, nothing
else sharing its chip model plugged in at the same time.

The only way to change which vid/pid the launcher uses is to overwrite
usb_config.json (by re-running this script, or editing it by hand) — there
is no env var or CLI override, so nothing here can drift out of sync with
what's actually plugged in.

Usage:
    python detect_usb_config.py
"""

from __future__ import annotations

import json
import sys

from serial.tools import list_ports

from pi_demo_common import REPO_ROOT

USB_CONFIG_PATH = REPO_ROOT / "tools" / "pi_demo" / "usb_config.json"


def main(argv: list[str] | None = None) -> int:
    ports = list_ports.comports()
    if not ports:
        print("No USB-serial devices found. Plug in the Nano and try again.",
              file=sys.stderr)
        return 1
    if len(ports) > 1:
        print("Multiple USB-serial devices found — unplug everything except "
              "the Nano and try again:", file=sys.stderr)
        for p in ports:
            print(f"  {p.device}  vid={p.vid:04X} pid={p.pid:04X}"
                  if p.vid is not None else f"  {p.device}  (no vid/pid)",
                  file=sys.stderr)
        return 1

    port = ports[0]
    if port.vid is None or port.pid is None:
        print(f"{port.device} doesn't report a USB vid/pid — can't use it "
              "for auto-discovery.", file=sys.stderr)
        return 1

    config = {"vid": f"0x{port.vid:04X}", "pid": f"0x{port.pid:04X}"}
    USB_CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Detected {port.device}: vid={config['vid']} pid={config['pid']}")
    print(f"Wrote {USB_CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
