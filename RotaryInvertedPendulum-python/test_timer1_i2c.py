"""
Lit DiagTimer1I2C (2Mbaud, texte ASCII).
Affiche rawAngle AVANT et APRES engine.init(), puis le loop() pendant 5s.
Lance apres power cycle (debranche/rebranche USB, attends 5s).
Sauvegarde dans test_timer1_i2c.log
"""
import serial, time, os

LOG = os.path.join(os.path.dirname(__file__), 'test_timer1_i2c.log')
log_f = open(LOG, 'w', encoding='utf-8')

def p(msg=''):
    print(msg, flush=True)
    print(msg, file=log_f, flush=True)

ser = serial.Serial('COM3', 2_000_000, timeout=0.5)
time.sleep(2.5)
ser.reset_input_buffer()

p("=== DiagTimer1I2C ===")
t0 = time.time()
while time.time() - t0 < 8:
    line = ser.readline().decode('ascii', errors='replace').strip()
    if line:
        p(f"{time.time()-t0:5.2f}s  {line}")

ser.close()
p(f"\nLog : {LOG}")
log_f.close()
