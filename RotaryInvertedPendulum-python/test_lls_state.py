"""
Test LowLevelServer — pendule libre.
Lance ce script toi-meme : il affiche les instructions en temps reel.
Sortie sauvegardee dans test_lls_state.log
"""
import serial, struct, time, math, sys, os

PORT = 'COM4'
BAUD = 2_000_000
LOG  = os.path.join(os.path.dirname(__file__), 'test_lls_state.log')

log_f = open(LOG, 'w', encoding='utf-8')

def p(msg=''):
    print(msg, flush=True)
    print(msg, file=log_f, flush=True)

def get_state(ser):
    ser.write(b'\x02')
    ser.flush()
    data = ser.read(20)
    if len(data) == 20:
        return struct.unpack('<Iffff', data)
    return None

# --- Connexion ---
p("Connexion COM4 (dtr=False, pas de reset Arduino)...")
ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 1.0
ser.dtr = False
ser.open()
time.sleep(3.0)
ser.reset_input_buffer()

# --- READY ---
p("Envoi READY...")
ser.write(b'\x01')
ser.flush()
resp = ser.read(1)
if resp != b'\x01':
    p(f"READY FAILED: {resp.hex() if resp else 'timeout'}")
    ser.close(); log_f.close(); sys.exit(1)
p("READY OK\n")

# --- Phase 1 : immobile ---
p("=" * 50)
p("PHASE 1 — LAISSE LE PENDULE IMMOBILE")
p("=" * 50)
p(f"{'i':>3}  {'t(s)':>7}  {'pen(deg)':>9}  {'pen_vel':>9}")
p("-" * 40)
pen_vals = []
for i in range(10):
    s = get_state(ser)
    if s:
        t_us, m, p_pos, mv, pv = s
        pen_vals.append(p_pos)
        p(f"{i:3d}  {t_us/1e6:7.3f}  {math.degrees(p_pos):9.2f}  {pv:9.3f}")
    else:
        p(f"{i:3d}  TIMEOUT")
    time.sleep(0.2)

# --- Countdown avant mouvement ---
p()
p("Preparation...")
for n in [3, 2, 1]:
    p(f"  {n}...")
    time.sleep(1.0)

p()
p("=" * 50)
p(">>>  BOUGE LE PENDULE MAINTENANT — 5 secondes  <<<")
p("=" * 50)

# --- Phase 2 : mouvement ---
for i in range(25):
    s = get_state(ser)
    if s:
        t_us, m, p_pos, mv, pv = s
        pen_vals.append(p_pos)
        p(f"{10+i:3d}  {t_us/1e6:7.3f}  {math.degrees(p_pos):9.2f}  {pv:9.3f}")
    else:
        p(f"{10+i:3d}  TIMEOUT")
    time.sleep(0.2)

p()
p("=" * 50)
p(">>>  STOP — arrête le pendule  <<<")
p("=" * 50)

# --- Phase 3 : retour immobile ---
p("Retour immobile (5 lectures)...")
for i in range(5):
    s = get_state(ser)
    if s:
        t_us, m, p_pos, mv, pv = s
        pen_vals.append(p_pos)
        p(f"{35+i:3d}  {t_us/1e6:7.3f}  {math.degrees(p_pos):9.2f}  {pv:9.3f}")
    else:
        p(f"{35+i:3d}  TIMEOUT")
    time.sleep(0.2)

ser.close()

# --- Bilan ---
p()
p("=" * 50)
p("BILAN")
p("=" * 50)
if pen_vals:
    lo = math.degrees(min(pen_vals))
    hi = math.degrees(max(pen_vals))
    spread = hi - lo
    p(f"Plage : {lo:.2f} deg .. {hi:.2f} deg")
    p(f"Spread: {spread:.2f} deg")
    if spread > 5.0:
        p("OK  : pen_pos varie — AS5600 + accumulation fonctionnent")
    elif spread > 0.1:
        p("PARTIEL : legere variation, verifier alignement aimant/capteur")
    else:
        p("ATTENTION : pen_pos ne varie pas — aimant trop loin ou AS5600 bloque")
p(f"\nLog sauvegarde : {LOG}")
log_f.close()
