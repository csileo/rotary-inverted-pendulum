"""
Lit TestEncoder (500kbaud, texte) pendant 8 secondes.
Sauvegarde dans test_encoder_log.log
Bouge le pendule quand tu vois BOUGE, arrête quand tu vois STOP.
"""
import serial, time, os

LOG = os.path.join(os.path.dirname(__file__), 'test_encoder_log.log')
log_f = open(LOG, 'w', encoding='utf-8')

def p(msg=''):
    print(msg, flush=True)
    print(msg, file=log_f, flush=True)

ser = serial.Serial('COM3', 500000, timeout=0.5)
time.sleep(2.5)
ser.reset_input_buffer()

p("=== TestEncoder - lecture 8s ===")
p()

t0 = time.time()
phase = 'immobile'
moved = False

while time.time() - t0 < 10:
    elapsed = time.time() - t0

    if elapsed > 2 and phase == 'immobile':
        phase = 'move'
        p()
        p(">>> BOUGE LE PENDULE MAINTENANT <<<")
        p()

    if elapsed > 7 and phase == 'move':
        phase = 'stop'
        p()
        p(">>> STOP <<<")
        p()

    line = ser.readline().decode('ascii', errors='replace').strip()
    if line:
        p(f"{elapsed:5.1f}s  {line}")

ser.close()
p()
p(f"Log : {LOG}")
log_f.close()
