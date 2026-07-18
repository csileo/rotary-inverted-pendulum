"""
Affiche la vitesse et l'ETA d'un run en cours.

Usage:
    python eta.py <run_name> <total_steps>
    python eta.py curriculum7_stage1 500000
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python eta.py <run_name> <total_steps>")
        sys.exit(1)

    run_name = sys.argv[1]
    total_steps = int(sys.argv[2])

    monitor = RUNS / run_name / "monitor.monitor.csv"
    if not monitor.exists():
        print(f"Fichier introuvable : {monitor}")
        sys.exit(1)

    lines = [l for l in monitor.read_text().splitlines() if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    if len(rows) < 2:
        print("Pas assez d'episodes pour estimer la vitesse.")
        sys.exit(1)

    t0 = float(rows[0]["t"])
    t1 = float(rows[-1]["t"])
    elapsed = t1 - t0
    steps_done = sum(int(r["l"]) for r in rows)
    speed = steps_done / elapsed if elapsed > 0 else 0

    # Approximation du step courant (monitor ne log que les steps d'episode)
    current = steps_done
    remaining = max(0, total_steps - current)
    eta_s = remaining / speed if speed > 0 else 0

    print(f"Run      : {run_name}")
    print(f"Episodes : {len(rows)}")
    print(f"Steps    : ~{current:,} / {total_steps:,}  ({100*current/total_steps:.1f}%)")
    print(f"Vitesse  : {speed:.1f} steps/s")
    print(f"ETA      : {eta_s/3600:.1f}h  ({eta_s/60:.0f} min)")


if __name__ == "__main__":
    main()
