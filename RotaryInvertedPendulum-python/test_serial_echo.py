"""
Test minimal : envoie 3 octets a 2Mbaud, verifie l'echo.
TestSerial2M.ino echo chaque octet recu.
"""
import serial, time

ser = serial.Serial('COM3', 2_000_000, timeout=1.0)
time.sleep(2.0)
ser.reset_input_buffer()

for byte in [0x41, 0x42, 0x43]:  # 'A', 'B', 'C'
    ser.write(bytes([byte]))
    ser.flush()
    resp = ser.read(1)
    ok = resp == bytes([byte])
    print(f"Envoi 0x{byte:02X} -> recu {resp.hex() if resp else 'timeout'} {'OK' if ok else 'FAIL'}")

ser.close()
