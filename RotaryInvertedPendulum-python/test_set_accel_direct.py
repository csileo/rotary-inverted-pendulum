"""
Test direct de CMD_SET_ACCEL sur LowLevelServer, sans passer par
accel_step_probe.py ni run_policy.py — élimine toute pause > quelques ms
entre les commandes (donc le watchdog 200 ms ne peut pas se déclencher)
et n'utilise que le protocole binaire brut, pour isoler si le firmware
LowLevelServer complet (I2C + Timer1 + protocole série réel) répond à
CMD_SET_ACCEL quand rien d'autre n'interfère avec le timing des commandes.

Lance ce script toi-meme, ajuste PORT si besoin.
"""
import serial, struct, time, sys

PORT = 'COM4'
BAUD = 2_000_000
ACCEL = 50.0       # rad/s^2
N_TICKS = 150      # ~ quelques secondes vu la latence de chaque aller-retour

CMD_READY = 0x01
CMD_GET_STATE = 0x02
CMD_SET_ACCEL = 0x03
CMD_ENGAGE_MOTOR = 0x04
CMD_DISENGAGE_MOTOR = 0x05


def get_state(ser):
    ser.write(bytes([CMD_GET_STATE]))
    data = ser.read(20)
    if len(data) != 20:
        return None
    return struct.unpack('<Iffff', data)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    ser = serial.Serial(port, BAUD, timeout=0.5)
    print(f"Connexion {port} @ {BAUD}...")
    time.sleep(2.5)  # reset bootloader + setup()
    ser.reset_input_buffer()

    ser.write(bytes([CMD_READY]))
    resp = ser.read(1)
    if resp != bytes([CMD_READY]):
        print(f"READY FAILED: {resp!r}")
        ser.close()
        sys.exit(1)
    print("READY OK")

    ser.write(bytes([CMD_ENGAGE_MOTOR]))
    print("ENGAGE envoyé")
    time.sleep(0.05)

    payload = bytes([CMD_SET_ACCEL]) + struct.pack('<f', ACCEL)

    t0 = time.monotonic()
    pos0 = None
    for i in range(N_TICKS):
        ser.write(payload)
        s = get_state(ser)
        if s is None:
            print(f"{i:3d}  TIMEOUT on get_state")
            continue
        t_us, motor_pos, pen_pos, motor_vel, pen_vel = s
        if pos0 is None:
            pos0 = motor_pos
        if i % 10 == 0:
            print(f"{i:3d}  t={time.monotonic()-t0:5.2f}s  "
                  f"motor_pos={motor_pos:+.4f}  motor_vel={motor_vel:+.4f}  "
                  f"delta_pos={motor_pos - pos0:+.4f}")

    # Stop proprement.
    ser.write(bytes([CMD_SET_ACCEL]) + struct.pack('<f', 0.0))
    time.sleep(0.05)
    ser.write(bytes([CMD_DISENGAGE_MOTOR]))
    ser.close()
    print("Terminé.")


if __name__ == "__main__":
    main()
