# Setting up a Raspberry Pi 3B+ for the unattended demo

Step-by-step guide to turn a brand-new Raspberry Pi 3B+ into the piece
that flashes the Nano, waits for 12V and USB to be plugged in (in any
order), and starts the balance policy — no screen or keyboard needed once
it's set up. That runtime behavior is `tools/pi_demo/`'s job (see
[`tools/pi_demo/README.md`](../tools/pi_demo/README.md)); this document
covers everything *before* that: preparing the SD card, installing
dependencies on the Pi, and wiring the rig physically.

## 1. Hardware needed

- Raspberry Pi 3B+
- micro SD card ≥ 8 GB (16-32 GB recommended), class 10
- Official Pi power supply: 5V / 2.5A micro-USB (an undersized supply
  causes random reboots under load — the #1 cause of ghost bugs on a Pi
  3B+)
- USB cable to connect the Pi to the Nano (same connector as pictured in
  `docs/BOM.md` — Micro-USB or USB-C on the Nano side depending on the
  clone)
- The assembled rig with its 12V power supply (see `docs/BOM.md` and
  `docs/electronics_design.md`)
- A computer to flash the SD card (Windows/macOS/Linux, doesn't matter)
- Optional: HDMI screen + keyboard for the first boot (otherwise
  everything is done headless over SSH, see step 2)

## 2. Flash the SD card

1. Install [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
   on the computer.
2. Insert the micro SD card.
3. In Raspberry Pi Imager:
   - **Device**: Raspberry Pi 3
   - **OS**: *Raspberry Pi OS Lite (64-bit)* — no need for a desktop
     environment since the Pi will run headless; it also leaves more RAM
     free for PyTorch/stable-baselines3.
   - **Storage**: the inserted SD card
4. Click the gear icon (⚙️, "Edit Settings" / advanced options) **before**
   writing the image, and configure:
   - Hostname (e.g. `pendulum-pi`)
   - Enable SSH → "Use password authentication" (or a public key if you
     have one)
   - Username + password
   - Wi-Fi (SSID + password + country) if the Pi isn't on Ethernet
   - Timezone / keyboard layout
5. Write the image, wait for verification, eject the card.

These settings avoid any physical screen/keyboard: the Pi boots straight
into SSH on the right network.

## 3. First boot and connection

1. Insert the SD card into the Pi, plug in the official power supply
   (not the rig yet).
2. Wait ~1-2 min for the first boot (filesystem expansion, automatic
   reboot).
3. Connect:
   ```bash
   ssh <username>@pendulum-pi.local
   ```
   (or the IP directly if `.local`/mDNS doesn't resolve on your network).
4. Update the system:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

## 4. System dependencies

```bash
sudo apt install -y git python3-venv python3-pip curl
```

**arduino-cli** (needed to flash the Nano from the Pi):

```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
sudo mv bin/arduino-cli /usr/local/bin/
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli lib install FastAccelStepper AS5600
```

Confirm the Nano is actually seen once plugged in over USB (step 7) with
`arduino-cli board list`.

## 5. Clone the repository

For a Pi dedicated purely to running the demo (no training), the `demo`
branch is enough — it only carries the firmware, the reference policies,
and `tools/pi_demo/`:

```bash
git clone --branch demo --single-branch \
    https://github.com/csileo/rotary-inverted-pendulum.git
cd rotary-inverted-pendulum
```

## 6. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On a Pi 3B+ (ARM, no GPU), `stable-baselines3` pulls in PyTorch CPU-only
automatically — this can take several minutes, that's expected.

If `pip install` fails partway through the PyTorch download with `OSError:
[Errno 28] No space left on device`, it's very likely not the SD card itself:
`/tmp` on Raspberry Pi OS is often a `tmpfs` (RAM-backed) mount sized at
~50% of RAM, e.g. ~450 MB on a Pi 3B+'s 1 GB — too small for the ~430 MB
PyTorch wheel pip downloads/unpacks there. Confirm with `mount | grep /tmp`,
then point pip at a temp dir on the SD card instead:

```bash
mkdir -p ~/pip_tmp
TMPDIR=~/pip_tmp pip install -r requirements.txt
```

## 7. Plug in the Nano and set up the AS5600 module

1. Plug the Nano into the Pi over USB.
2. `RotaryInvertedPendulum-arduino/LowLevelServer/hw_config.h` is
   deliberately absent from the repo (no safe default — see CLAUDE.md):
   copy the profile matching the AS5600 module mounted on this rig from
   `RotaryInvertedPendulum-arduino/LowLevelServer/hw_profiles/`:
   ```bash
   cp RotaryInvertedPendulum-arduino/LowLevelServer/hw_profiles/as5600_hailege_clone.h \
      RotaryInvertedPendulum-arduino/LowLevelServer/hw_config.h
   # or as5600_seeed.h for an original Seeed module — see docs/BOM.md
   ```
3. Detect the Nano for this specific Pi (do this once, or again if the
   Nano is swapped for one with a different USB-serial chip) — **unplug
   every other USB-serial device before running this**, the script
   assumes exactly one serial device is connected:
   ```bash
   cd tools/pi_demo
   python detect_usb_config.py
   cd ../..
   ```

## 8. Manual test before automating

Before automating anything, confirm the full chain works by hand, with
the rig powered at 12V and the Nano on USB:

```bash
cd RotaryInvertedPendulum-python/src/rl
python run_policy.py --policy models/policy_working_balance.zip \
    --frame-stack 3 --duration-s 30 --port <PORT>
```

`<PORT>`: found with `arduino-cli board list` (typically `/dev/ttyUSB0`
on a Pi). If it compiles/flashes and balances, everything is in place;
move on to automation.

A distilled student MLP is also available as a lighter-weight alternative
to the SAC teacher `.zip` — same balance quality (upright ≈ 0.987 on a Pi
3B+, validated 2026-07-28), much smaller and faster to load:

```bash
python run_policy.py \
    --policy models/distill_working_balance_h32_dagger/student.pt \
    --frame-stack 3 --duration-s 30 --port <PORT>
```

Both checkpoints are inference-only here — `run_policy.py` still runs on
the Pi itself, not on the Nano; see `models/README.md` for the distinction
and why an actual on-device (standalone) deployment of the student was not
pursued.

## 9. Automate: `tools/pi_demo/run_demo.py`

This is the script that makes the demo tolerant to plug-in order — Pi
power, the pendulum's 12V, and the Nano's USB cable can be plugged in any
order and with any delay between them; it waits out each precondition
instead of failing on the first one missing (see
`tools/pi_demo/README.md`).

Run it once by hand to check:

```bash
cd tools/pi_demo
python run_demo.py
```

Then wrap it in a `systemd` service so it runs automatically on every Pi
boot (e.g. after a power outage):

```bash
sudo tee /etc/systemd/system/pendulum-demo.service > /dev/null <<'EOF'
[Unit]
Description=Rotary inverted pendulum - unattended demo
After=network.target

[Service]
Type=simple
User=<username>
WorkingDirectory=/home/<username>/rotary-inverted-pendulum/tools/pi_demo
ExecStart=/home/<username>/rotary-inverted-pendulum/.venv/bin/python run_demo.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now pendulum-demo.service
```

Replace `<username>` with the username configured in step 2.
`run_demo.py` blocks until the policy finishes (`--duration-s`),
Ctrl-C/SIGTERM, or a wait step times out — `Restart=on-failure` cleanly
restarts the wait cycle if needed (e.g. after the Nano gets unplugged).

Optional environment variables (add under `[Service]` with
`Environment=`, see `tools/pi_demo/README.md`):

| Variable | Meaning |
|---|---|
| `PENDULUM_POLICY` | Path to the `.zip`/`.pt` checkpoint to load. Defaults to `models/distill_working_balance_h32_dagger/student.pt`; set to `models/policy_working_balance.zip` for the full SAC teacher instead |
| `PENDULUM_FRAME_STACK` | Must match the checkpoint's training frame-stack |
| `PENDULUM_DURATION_S` | How long to balance before stopping |
| `PENDULUM_MOTOR_POWER_TIMEOUT_S` | How long to wait for 12V before giving up |

## 10. Physical mounting on the rig

1. Mount the Pi near the rig (e.g. under the base), away from motor
   vibration and cables that could snag the arm.
2. USB cable: Pi → Nano. Long enough not to constrain the arm's
   rotation.
3. Pi power: **separate** from the 12V motor rail — a dedicated
   micro-USB 5V/2.5A wall adapter, on a different outlet or a fixed power
   strip (no loose connection that could pull on the rig's wiring).
4. Pendulum 12V supply: unchanged, see `docs/electronics_design.md` and
   `docs/BOM.md`.
5. Final test: cut all power, then plug everything back in in a
   completely arbitrary order (12V first, or USB first, or Pi first,
   with varying delays between each). The service should wait patiently
   and start balancing as soon as everything is present — watch the logs
   with:
   ```bash
   journalctl -u pendulum-demo.service -f
   ```

## 11. Troubleshooting

- **`arduino-cli board list` doesn't see the Nano**: faulty USB cable
  (common with "charge only" cables), or a USB-serial chip (CH340 etc.)
  without a driver — rare on Raspberry Pi OS, which already ships the
  usual drivers.
- **`flash_if_needed.py` refuses to compile**: `hw_config.h` is missing —
  go back to step 7.
- **"Timed out waiting for motor power"**: check that the 12V adapter is
  plugged in and the rig's power switch (if present, see BOM) is on;
  otherwise a driver/Vref/enable wiring fault — see
  `docs/electronics_design.md`.
- **The service keeps restarting**: `journalctl -u
  pendulum-demo.service -f` for the exact error; most often a wrong
  venv/policy path in the service file.
- **Flaky USB detection after swapping the Nano**: rerun
  `detect_usb_config.py` (step 7) with only one serial device plugged in.
