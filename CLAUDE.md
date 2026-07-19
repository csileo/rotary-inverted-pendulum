# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active initiatives

- **RL controller**: a multi-phase effort to replace the hand-tuned PID with a learned swing-up + balance policy. The living plan, phase status, and decisions log are in `RL_PLAN.md` at the repo root. Read that file before working on anything under `RotaryInvertedPendulum-arduino/LowLevelServer/`, `RotaryInvertedPendulum-arduino/RLControl/`, or `RotaryInvertedPendulum-python/src/rl/`. Companion docs:
  - `docs/rl_transitions.md` — the `(s, a, r, s')` transition contract in plain English.
  - `docs/async_control_architecture.md` — the threaded runtime that holds the configured control rate strictly during fine-tuning.
  - `docs/control_rate_selection.md` — how to pick `control_freq_hz` and `max_action_delta_rad` from sysid measurements.
  - `docs/sysid_runbook.md` — the measurement procedure for the inputs the two docs above depend on.

## Where physical parameters live

The rotary inverted pendulum has one source of truth per parameter
class. Updating a number in any other place is a bug — the chain is:

- **Pendulum body geometry** (mass, COM, inertia tensor): authored in
  Onshape → exported to `urdf/model.urdf` → parsed at import time by
  `RotaryInvertedPendulum-python/src/rl/pendulum_geometry.py` →
  consumed by `pendulum_env.py` (MuJoCo sim + DR), `sysid_core.py`
  (friction derivation, sanity-check against measured period), and the
  Julia stack (MeshCat viz, RigidBodyDynamics MPC). To change pendulum
  mass/COM/inertia, edit Onshape → export the URDF; nothing else.

- **Per-rig dynamic state** (viscous + Coulomb friction): measured by
  `sysid_wizard.py` from a free-swing recording on the actual hardware,
  written to `RotaryInvertedPendulum-python/src/rl/sysid_params.json`,
  loaded into `PendulumParams` alongside the URDF constants. These do
  vary between rebuilds (bearings, grease, temperature) and are the
  only quantities the sysid pipeline measures.

- **Arm geometry** (length, mass, COM): currently hard-coded constants
  in `pendulum_env.py` (`ARM_*`). When CAD-validated, will follow the
  pendulum pattern — read from the `arm` link in `urdf/model.urdf`.

- **Hardware/firmware constants** (motor max accel, AS5600 resolution,
  hard-stop limits): module constants in `pendulum_env.py`, with the
  Arduino firmware as the upstream source for the motor-side values.

## Overview

This is a rotary inverted pendulum control project - a classic control theory problem demonstrating system stabilization. The system consists of a pendulum mounted on a rotary base that must be balanced upright by controlling the base rotation. The project includes mechanical designs (3D printable), electronics (Arduino-based), and multiple control implementations.

## Project Structure

The repository is organized into three main software components:

- **RotaryInvertedPendulum-arduino/**: Arduino C++ code for the microcontroller (Arduino Nano)
  - `LowLevelServer/`: Low-level server for computer-controlled operation via serial
  - `PIDControl/`: Self-contained PID controller running on Arduino
  - `TestEncoder/`, `TestMotor/`, `TestSerial/`, etc.: Hardware test sketches

- **RotaryInvertedPendulum-julia/**: Julia control algorithms and visualization
  - `src/`: Main Julia package code
  - `notebooks/`: Jupyter notebooks for experimentation (e.g., MPC development)

- **RotaryInvertedPendulum-python/**: Python control implementations
  - `src/`: Python control code (gamepad control)
  - `test/`: Test scripts

Additional directories:
- `meshes/`: 3D-printable STL files for mechanical components
- `diagrams/`: Circuit diagrams and system schematics
- `urdf/`: Robot model description files

## Hardware Architecture

The system uses:
- **Arduino Nano**: Microcontroller for sensor reading and motor control
- **AS5600 Magnetic Encoder**: Measures pendulum angle (I2C communication)
- **Stepper Motor (NEMA17)**: Rotates the base arm (via driver like DRV8825/A4988/TMC2209)
- **AccelStepper Library**: Controls stepper motor with acceleration profiles

Communication between Arduino and computer is via serial at 2,000,000 baud.

## Control Approaches

Two main control architectures:

1. **On-device control** (Arduino): PID controller runs entirely on Arduino Nano
   - Portable, no computer required
   - Limited computational power
   - See: `RotaryInvertedPendulum-arduino/PIDControl/PIDControl.ino`

2. **Computer-based control** (Julia/Python): Arduino acts as low-level server
   - High computational power for advanced algorithms (MPC, LQR)
   - Requires USB connection to computer
   - Arduino code: `LowLevelServer/LowLevelServer.ino`
   - Client code: Julia files in `src/` or Python in `RotaryInvertedPendulum-python/`

## Serial Communication Protocol

### Text-based protocol (PIDControl)
Commands are sent as text strings:
- `"1"`: Check if ready
- `"2"`: Get motor position
- `"3"`: Get pendulum position
- `"4 <position>"`: Set target motor position
- `"5"`: Start motor
- `"6"`: Stop motor

### Binary protocol (LowLevelServer)
Commands are single bytes:
- `0x01`: Check ready
- `0x02`: Get state (returns time, motor position, pendulum position as floats)
- `0x03`: Set target (expects 4-byte float in radians)
- `0x04`: Engage motor
- `0x05`: Disengage motor

## Julia Development

### Setup
```bash
cd RotaryInvertedPendulum-julia
julia --project=.
```

In Julia REPL:
```julia
using Pkg
Pkg.instantiate()  # Install dependencies
```

### Running Control Scripts

PID control from Julia:
```julia
using RotaryInvertedPendulum
pid_control()  # Default: 2000000 baud, 200 Hz control
```

Low-level server client with visualization:
```bash
julia --project=. ../RotaryInvertedPendulum-arduino/LowLevelServer/client.jl --visualise
```

### Key Julia Files

- `RotaryInvertedPendulum.jl`: Main module, defines serial commands and Arduino communication
- `control_pid.jl`: PID controller implementation communicating over serial
- `control_gamepad.jl`: Gamepad-based manual control
- `mpc.jl`: Model Predictive Control implementation with system linearization
- `utils.jl`: Utility functions
- `precompile.jl`: Package precompilation for faster startup

### Dependencies
The Julia package uses:
- `LibSerialPort`: Serial communication with Arduino
- `RigidBodyDynamics`: For system dynamics modeling
- `ForwardDiff`: Automatic differentiation for MPC linearization
- `MeshCat`, `MeshCatMechanisms`: 3D visualization
- `Joysticks`: Gamepad support
- `Plots`: Data plotting

## Arduino Development

### Prerequisites
Libraries required (install via Arduino IDE Library Manager):
- [AccelStepper](https://www.airspayce.com/mikem/arduino/AccelStepper/)
- [AS5600](https://github.com/Seeed-Studio/Seeed_Arduino_AS5600) (included in `libs/`)

### Flashing Arduino
1. Open `.ino` file in Arduino IDE
2. Select Board: "Arduino Nano"
3. Select Port: `/dev/cu.usbserial-*` (macOS) or appropriate COM port
4. Upload sketch

### Key Arduino Concepts

**Stepper Motor Configuration:**
- Microstepping: 8 (default) → 1600 steps/revolution
- Enable pin inverted (DRV8825 uses active-low enable)
- Max speed: 200,000 steps/sec
- Acceleration: 100,000 steps/sec²

**AS5600 Encoder:**
- Provides 12-bit resolution (0-4095 raw values)
- Maps to 0-360° or 0-2π radians
- Handles multi-revolution tracking with wraparound logic
- Check magnet strength on startup

**PID Control Parameters:**
- Control frequency: 200-1000 Hz (varies by implementation)
- Manually tuned gains in `PIDControl.ino`: Kp=2.2, Ki=1.6, Kd=0.005
- Motor position limits: ±90° from starting position (prevents wire choking)
- Engagement margin: ±25° from vertical

## Python Development

Python implementation is less developed but includes gamepad control and serial test scripts. Install dependencies and run scripts from `RotaryInvertedPendulum-python/`.

## Common Development Tasks

### Testing Hardware
- **Test encoder**: Flash `TestEncoder/TestEncoder.ino`, open Serial Monitor/Plotter
- **Test motor**: Flash `TestMotor/TestMotor.ino`
- **Test serial communication**: Flash `TestSerial/TestSerial.ino`

### Hardware-in-the-Loop Testing
For automated testing with hardware connected, use the serial monitoring script:

```bash
./RotaryInvertedPendulum-arduino/scripts/monitor_serial.sh <port> <baud_rate> <duration>
```

This script properly handles Arduino reset on serial connection and flushes old buffer data to provide clean output. Useful for verifying Arduino behavior during development without manual intervention.

Example:
```bash
./RotaryInvertedPendulum-arduino/scripts/monitor_serial.sh /dev/cu.usbserial-10 115200 10
```

**Note:** This approach avoids common issues with direct `cat` or `stty` usage that can cause double resets or capture stale buffered data.

### Serial Port Issues
On macOS, the Arduino typically appears as `/dev/cu.usbserial-110` or similar. Update the port string in Julia/Python code if different.

### Current Limiting
Set stepper driver current limit to 0.9A (90% of motor's 1A rating) using the onboard potentiometer. Vref formulas vary by driver - see README.md.

## Control Theory Notes

The system implements a state-space approach where:
- State: `[motor_angle, pendulum_angle, motor_velocity, pendulum_velocity]`
- Control input: motor torque (converted to position commands for stepper)

**MPC Implementation** (`mpc.jl`):
- Linearizes nonlinear dynamics using `RigidBodyDynamics` and `ForwardDiff`
- Uses RK4 integration for discrete-time dynamics
- Linearization point: pendulum upright (π radians), motor at origin

**State Machine** (both Arduino and Julia PID):
- `WAITING`: Motor disabled, waiting for pendulum near vertical
- `BALANCING`: Motor engaged, actively controlling

## URDF and Visualization

The `urdf/` directory contains robot description files used by `RigidBodyDynamics.jl` for dynamics computation and `MeshCat` for 3D visualization.

## Serial Port Configuration

Arduino baud rate: 2,000,000 (high-speed for real-time control)
- Read timeout: 50ms (typical)
- Write timeout: 10ms (typical)

Always call `wait_until_ready(arduino)` after opening serial connection to synchronize with Arduino.
