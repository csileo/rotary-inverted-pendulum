# Auto-demo launcher

Makes the "power it on, walk away, it balances" demo robust to whatever
order the person plugs in the Pi/computer's power, the pendulum's 12V
adapter, and the Nano's USB cable in — and to arbitrary delay between each.
Nothing here assumes a particular order; each step waits out its own
precondition instead of failing on the first miss.

Runs unchanged on Linux, macOS, and Windows: the Nano is found by scanning
USB devices for its vid/pid (via `pyserial`'s `serial.tools.list_ports`)
instead of relying on a fixed `/dev/ttyUSB0`-style path or a udev rule, and
the launcher itself is plain Python rather than a shell script.

- **`flash_if_needed.py`** — waits for the Nano to show up, then compares
  a hash the firmware reports (`CMD_GET_FIRMWARE_VERSION`, baked in at
  compile time by
  `RotaryInvertedPendulum-arduino/LowLevelServer/gen_firmware_version.py`)
  against the local sketch source. Only runs `arduino-cli compile
  --upload` on a mismatch (or no response at all) — skips the ~15-30s
  flash cycle on every normal boot where the Nano already has the right
  firmware. Refuses to compile at all if
  `RotaryInvertedPendulum-arduino/LowLevelServer/hw_config.h` is missing
  (see that directory's `hw_profiles/`) — there's no safe default AS5600
  backend to guess.
- **`check_motor_power.py`** — the rig has no voltage sensing on the 12V
  rail, so this checks it indirectly: issue a small, brief acceleration
  pulse and confirm the AS5600 actually saw the motor move. If it didn't,
  the 12V adapter isn't plugged in (or there's a driver/Vref/enable wiring
  fault) — no new sensing hardware required.
- **`run_demo.py`** — orchestrates the above in order, waiting (not
  failing) at each step until its precondition is met, then runs the
  policy. Loops forever (10s pause between cycles by default) so a
  headless/no-network Pi keeps re-demoing without anyone reconnecting to
  restart it — Ctrl-C or SIGTERM (`systemctl stop`) is the only way out.
  Loads the policy once and keeps one serial connection to the Nano open
  across every cycle (calling `run_policy.py`'s `run_control_loop()`
  in-process instead of launching it as a fresh subprocess each time) —
  opening a new connection resets the Arduino, so reusing one avoids
  paying that reset repeatedly. The connection is only rediscovered /
  reflashed / reopened if something actually goes wrong (e.g. the Nano
  gets unplugged). Defaults `PENDULUM_POLICY` to the `.npz` (NumPy-only)
  build of the distilled student, not the `.pt`/torch one, so this
  process never has to import torch at all — see `numpy_student.py`.

## Which Nano to talk to

`pi_demo_common.py` resolves the vid/pid to search for like this:

1. `usb_config.json` in this directory, if it exists — this machine's own
   detected chip, written by `detect_usb_config.py`.
2. Otherwise, `usb_profiles/ch340.json` — the tracked reference default
   (confirmed in `docs/BOM.md` for both known builds).

`usb_config.json` is gitignored: it's a fact about this specific machine's
USB-serial chip, not something to share across forks. **The only way to
change which chip is used is to overwrite `usb_config.json`** — there is
no environment variable or CLI flag override, so it can never silently
drift out of sync with what's actually plugged in.

If your Nano uses a different chip than CH340/CH341, either add a new
`usb_profiles/<name>.json` (same two-key format) and copy it to
`usb_config.json`, or just run `detect_usb_config.py` with only the Nano
plugged in — it detects whichever single USB-serial device is present and
writes `usb_config.json` directly.

## Setup

1. Install Python deps (repo root `requirements.txt`, or the `demo`
   branch's copy if this is a demo-only checkout) and make sure
   `arduino-cli` is on `PATH` for the flash step.
2. Run `python detect_usb_config.py` once, with only the Nano plugged in
   (skip this if your chip already matches `usb_profiles/ch340.json`).
3. Make sure `RotaryInvertedPendulum-arduino/LowLevelServer/hw_config.h`
   is set up for your AS5600 module (see that directory's README/comments).
4. Run it: `python run_demo.py`. Wire it to whatever "on demand" trigger
   you want — a systemd service on the Pi, a Scheduled Task on Windows, a
   physical button via a GPIO watcher, cron, etc. This repo doesn't
   prescribe one; the script is a plain foreground process that loops
   forever (see above), blocking until Ctrl-C/SIGTERM.

## Environment variables

All optional — see `run_demo.py` for the exact defaults. None of these
select which Nano to talk to (see above) — only what to run once it's found.

| Variable | Meaning |
|---|---|
| `PENDULUM_POLICY` | Path to the `.zip`/`.pt`/`.npz` checkpoint to run |
| `PENDULUM_FRAME_STACK` | Must match the checkpoint's training frame-stack |
| `PENDULUM_DURATION_S` | How long to balance before stopping (default 30s) |
| `PENDULUM_MOTOR_POWER_TIMEOUT_S` | How long to wait for 12V before giving up |
| `PENDULUM_LOOP_DELAY_S` | Pause between demo cycles (default 10s) |
| `PENDULUM_CONTROL_FREQ_HZ` | Control loop frequency, must match training (default 35.0) |
| `PENDULUM_MAX_ACCEL_RAD_S2` | Action-to-acceleration scale, must match training (default 150.0) |
