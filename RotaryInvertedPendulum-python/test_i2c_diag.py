"""
Envoie CMD_DEBUG_I2C (0x07) 20 fois et affiche le résultat brut de chaque lecture I2C.
Retour : [et_rc, rf_n, raw_hi, raw_lo]
  et_rc = 0 → endTransmission(false) OK ; sinon → adresse NACKée
  rf_n  = 2 → 2 octets lus         ; 0   → requestFrom a échoué
  raw   = valeur 12 bits AS5600 (0-4095)
"""
import serial, time, os

PORT = 'COM4'
BAUD = 2_000_000
LOG  = os.path.join(os.path.dirname(__file__), 'test_i2c_diag.log')

log_f = open(LOG, 'w', encoding='utf-8')

def p(msg=''):
    print(msg, flush=True)
    print(msg, file=log_f, flush=True)

p("=== test_i2c_diag ===")

ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 1.0
ser.dtr = False
ser.open()
time.sleep(3.0)
ser.reset_input_buffer()

# READY handshake
ser.write(b'\x01')
ser.flush()
resp = ser.read(1)
if resp != b'\x01':
    p(f"READY FAILED: {resp.hex() if resp else 'timeout'}")
    ser.close(); log_f.close(); exit(1)
p("READY OK\n")

p(f"{'#':>3}  {'et_rc':>6}  {'rf_n':>5}  {'raw':>5}  {'deg':>7}  note")
p("-" * 50)

for i in range(20):
    ser.write(b'\x07')
    ser.flush()
    data = ser.read(4)
    if len(data) == 4:
        et_rc, rf_n, hi, lo = data
        raw = (hi << 8) | lo
        deg = raw / 4096 * 360
        note = ''
        if et_rc != 0:
            note = f'← endTransmission NACK (rc={et_rc})'
        elif rf_n != 2:
            note = f'← requestFrom retourné {rf_n} (attendu 2)'
        elif raw == 0:
            note = '← raw=0 (position 0 ou échec silencieux?)'
        p(f"{i:3d}  {et_rc:6d}  {rf_n:5d}  {raw:5d}  {deg:7.2f}°  {note}")
    else:
        p(f"{i:3d}  TIMEOUT (reçu {len(data)} octets)")
    time.sleep(0.1)

ser.close()
p(f"\nLog : {LOG}")
log_f.close()
