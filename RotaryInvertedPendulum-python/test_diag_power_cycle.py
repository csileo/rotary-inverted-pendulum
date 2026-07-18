"""
Test DiagLLS (I2C scan) apres power cycle propre (dtr=False = pas de reset supplementaire).
"""
import serial, time, sys

ser = serial.Serial()
ser.port = 'COM3'
ser.baudrate = 115200
ser.timeout = 0.5
ser.dtr = False   # pas de reset Arduino
ser.open()

time.sleep(0.1)
ser.reset_input_buffer()
print("Envoi trigger...")
ser.write(b'G')
ser.flush()

t0 = time.time()
while time.time() - t0 < 10:
    line = ser.readline().decode('ascii', errors='replace').strip()
    if line:
        print(f'{time.time()-t0:.2f}s: {line}')
        sys.stdout.flush()

ser.close()
print("FIN")
