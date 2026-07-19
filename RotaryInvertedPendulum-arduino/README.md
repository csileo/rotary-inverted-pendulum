# Arduino Sketches

This folder contains Arduino sketches for the Rotary Inverted Pendulum project.

## Prerequisites

Install [arduino-cli](https://arduino.github.io/arduino-cli/latest/installation/) and the AVR core:

```bash
arduino-cli core install arduino:avr
```

## Flashing Sketches

1. Find your Arduino's serial port:
   ```bash
   arduino-cli board list
   ```

2. Compile and flash (from repository root):
   ```bash
   arduino-cli compile --upload -p <PORT> --fqbn arduino:avr:nano:cpu=atmega328 RotaryInvertedPendulum-arduino/<SketchName>
   ```

   **Note:** This project uses `cpu=atmega328` (new bootloader). Do not use `cpu=atmega328old`.

## Sketches

### Test Sketches

These sketches are for testing individual hardware components.

#### TestHeartbeat

Blinks the LED in a double-pulse pattern. Useful for verifying the Arduino is powered and running.

- **Pattern:** ON(100ms)-OFF(100ms)-ON(100ms)-OFF(1000ms), repeating
- **Serial (115200):** Shows heartbeat count

**Use cases:** Quick hardware verification, power testing, bootloader check, debugging baseline.

#### TestEncoder

Tests the AS5600 magnetic encoder with multi-revolution tracking.

- **LED on** during setup, waits for magnet detection
- **LED flashes** when outputting readings
- **Serial (115200):** Outputs `pendulum_deg:value` (compatible with Serial Plotter)

**Troubleshooting:**
- "Waiting for magnet..." → Ensure magnet is positioned near the AS5600 sensor
- "Magnet strength too weak/strong" → Adjust magnet distance (~1-2mm)

#### TestMotor

Tests the stepper motor by oscillating between +90° and -90° positions.

- **LED on** during setup and during movement
- **Serial (115200):** Shows movement status
- Max speed set to 20,000 steps/sec (10x slower than production for visibility)

#### TestSerial

Measures serial round-trip time by echoing bytes.

- **LED on** during setup
- **LED flashes** on each received byte

Run the Julia measurement script:
```bash
julia --project=./RotaryInvertedPendulum-julia ./RotaryInvertedPendulum-arduino/TestSerial/measure_serial_rtt.jl <PORT> 115200
```

Expected output: ~2.5ms RTT, ~400 Hz max theoretical frequency.

#### TestServer

Simple server that tracks sine/cosine waves and responds to 'S' or 'C' byte requests.

```bash
julia --project=./RotaryInvertedPendulum-julia ./RotaryInvertedPendulum-arduino/TestServer/client.jl
```

### Production Sketches

#### LowLevelServer

Low-level server for computer-controlled operation. Arduino acts as a slave, receiving commands over serial (2,000,000 baud) from Julia/Python control code.

```bash
julia --project=./RotaryInvertedPendulum-julia ./RotaryInvertedPendulum-arduino/LowLevelServer/client.jl --visualise
```

#### PIDControl

Self-contained PID controller running entirely on Arduino. No computer required after flashing.

- **Serial (500000 baud):** Connect to view diagnostics and collect data
- **LED blink rate:** Fast (100ms) = waiting, Slow (500ms) = data output enabled

**Serial commands:**
- `P` - Toggle data output (CSV format at 100 Hz)
- `M` - Show magnet status
- `R` - Reset PID state

**Data collection:**
```bash
julia --project=./RotaryInvertedPendulum-julia ./RotaryInvertedPendulum-arduino/PIDControl/collect_and_plot.jl <PORT> [DURATION]
```

Example:
```bash
julia --project=./RotaryInvertedPendulum-julia ./RotaryInvertedPendulum-arduino/PIDControl/collect_and_plot.jl /dev/cu.usbserial-10 10
```

This collects data for the specified duration, generates plots, and saves CSV to `PIDControl/experiments/`.

See `PIDControl/TUNING_HISTORY.md` for tuning notes and `PIDControl/PIDControl.ino` header for architecture details.

## Hardware-in-the-Loop Testing

For automated testing with hardware connected:

```bash
./RotaryInvertedPendulum-arduino/scripts/monitor_serial.sh <PORT> <BAUD> <DURATION>
```

Example:
```bash
./RotaryInvertedPendulum-arduino/scripts/monitor_serial.sh /dev/cu.usbserial-10 115200 10
```
