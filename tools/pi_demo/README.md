# Auto-demo launcher

Makes the "power it on, walk away, it balances" demo robust to whatever
order the person plugs in the Pi/computer's power, the pendulum's 12V
adapter, and the Nano's USB cable in — and to arbitrary delay between each.
Nothing here assumes a particular order; each step waits out its own
precondition instead of failing on the first miss.

Runs unchanged on Linux, macOS, and Windows: the Nano is found by scanning
USB devices for its VID:PID (via `pyserial`'s `serial.tools.list_ports`)
instead of relying on a fixed `/dev/ttyUSB0`-style path or a udev rule, and
the launcher itself is plain Python rather than a shell script.

- **`flash_if_needed.py`** — waits for the Nano to show up, then compares
  a hash the firmware reports (`CMD_GET_FIRMWARE_VERSION`, baked in at
  compile time by
  `RotaryInvertedPendulum-arduino/LowLevelServer/gen_firmware_version.py`)
  against the local sketch source. Only runs `arduino-cli compile
  --upload` on a mismatch (or no response at all) — skips the ~15-30s
  flash cycle on every normal boot where the Nano already has the right
  firmware.
- **`check_motor_power.py`** — the rig has no voltage sensing on the 12V
  rail, so this checks it indirectly: issue a small, brief acceleration
  pulse and confirm the AS5600 actually saw the motor move. If it didn't,
  the 12V adapter isn't plugged in (or there's a driver/Vref/enable wiring
  fault) — no new sensing hardware required.
- **`run_demo.py`** — orchestrates the above in order, waiting (not
  failing) at each step until its precondition is met, then runs
  `run_policy.py`.

## Setup

1. Install Python deps (repo root `requirements.txt`, or the `demo`
   branch's copy if this is a demo-only checkout) and make sure
   `arduino-cli` is on `PATH` for the flash step.
2. If your Nano's USB-serial chip isn't the common CH340/CH341
   (`1A86:7523`, the default — see `pi_demo_common.py`), find its actual
   VID:PID on any OS with:
   ```
   python -m serial.tools.list_ports -v
   ```
   and pass it via `PENDULUM_USB_VID` / `PENDULUM_USB_PID` (see below), or
   `--vid`/`--pid` if running `flash_if_needed.py`/`check_motor_power.py`
   directly.
3. Run it: `python run_demo.py`. Wire it to whatever "on demand" trigger
   you want — a systemd service on the Pi, a Scheduled Task on Windows, a
   physical button via a GPIO watcher, cron, etc. This repo doesn't
   prescribe one; the script is a plain blocking foreground process that
   exits when the policy's `--duration-s` elapses, on Ctrl-C/SIGTERM, or
   if a wait step times out.

## Environment variables

All optional — see `run_demo.py` for the exact defaults.

| Variable | Meaning |
|---|---|
| `PENDULUM_PORT` | Serial device; skips USB auto-discovery if set |
| `PENDULUM_USB_VID` / `PENDULUM_USB_PID` | USB vendor/product ID (hex) used for auto-discovery |
| `PENDULUM_POLICY` | Path to the `.zip`/`.pt` checkpoint to run |
| `PENDULUM_FRAME_STACK` | Must match the checkpoint's training frame-stack |
| `PENDULUM_DURATION_S` | How long to balance before stopping |
| `PENDULUM_MOTOR_POWER_TIMEOUT_S` | How long to wait for 12V before giving up |
