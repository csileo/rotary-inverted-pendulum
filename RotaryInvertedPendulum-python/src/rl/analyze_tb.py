"""
Analyse les métriques TensorBoard d'un curriculum.

Usage:
    python analyze_tb.py curriculum4
    python analyze_tb.py curriculum4 --stages 500000 500000 300000

Gère deux cas :
  - n_envs=1 : tous les stages écrivent dans stage1/tb/ (bug TB connu),
    les steps étant continus on les sépare par --stages.
  - n_envs>1 : chaque stage a son propre dossier tb/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

METRICS = [
    ("train/ent_coef",      "ent_coef    "),
    ("rollout/ep_rew_mean", "ep_rew_mean "),
    ("train/actor_loss",    "actor_loss  "),
    ("train/critic_loss",   "critic_loss "),
]


def load_events(tb_dir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in tags}


def summarise(data: dict, step_lo: int, step_hi: int, label: str) -> None:
    print(f"\n{'='*52}")
    print(f"  {label}  (steps {step_lo:,} -> {step_hi:,})")
    print(f"{'='*52}")
    for tag, name in METRICS:
        pts = [(s, v) for s, v in data.get(tag, []) if step_lo <= s <= step_hi]
        if not pts:
            print(f"  {name}: n/a")
            continue
        first_v = pts[0][1]
        last_v  = pts[-1][1]
        min_v   = min(v for _, v in pts)
        max_v   = max(v for _, v in pts)
        arrow   = "v" if last_v < first_v else "^"
        print(f"  {name}: {first_v:+.3f} -> {last_v:+.3f}  {arrow}  "
              f"[min {min_v:.3f}  max {max_v:.3f}]  ({len(pts)} pts)")


def find_tb_dir(prefix: str, stage: int) -> Path | None:
    """Cherche le dossier tb/ pour un stage donné."""
    stage_dir = RUNS / f"{prefix}_stage{stage}"
    if not stage_dir.exists():
        return None
    tb = stage_dir / "tb"
    if not tb.exists():
        return None
    # Prend le sous-dossier SAC_N le plus récent
    subs = sorted(tb.iterdir(), key=lambda p: p.stat().st_mtime)
    return subs[-1] if subs else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prefix", help="préfixe du curriculum (ex: curriculum4)")
    p.add_argument("--stages", type=int, nargs="+", default=None,
                   help="steps par stage dans l'ordre (ex: 500000 500000 300000). "
                        "Nécessaire seulement si tous les stages partagent le même tb/.")
    args = p.parse_args()

    prefix = args.prefix

    # Découverte des stages
    stage_dirs = sorted(RUNS.glob(f"{prefix}_stage*"))
    n_stages = len(stage_dirs)
    if n_stages == 0:
        print(f"Aucun run trouvé pour le préfixe '{prefix}' dans {RUNS}")
        sys.exit(1)

    # Numéros de stage réels (ex: curriculum5 commence à stage2)
    stage_nums = sorted(
        int(d.name[len(prefix) + len("_stage"):]) for d in stage_dirs
    )
    first_stage = stage_nums[0]

    print(f"Curriculum : {prefix}  ({n_stages} stages détectés, stage{first_stage}..stage{stage_nums[-1]})")

    # Cas n_envs>1 : chaque stage a son propre tb/
    per_stage_tb = [find_tb_dir(prefix, n) for n in stage_nums]
    has_per_stage = all(d is not None for d in per_stage_tb)

    if has_per_stage:
        for n, tb_dir in zip(stage_nums, per_stage_tb):
            data = load_events(tb_dir)
            steps = sorted({s for tag in data for s, _ in data[tag]})
            lo, hi = (steps[0], steps[-1]) if steps else (0, 0)
            summarise(data, lo, hi, f"Stage {n}")
    else:
        # Cas n_envs=1 : tout est dans stage1/tb/
        tb_dir = find_tb_dir(prefix, first_stage)
        if tb_dir is None:
            print("Impossible de trouver le dossier tb/")
            sys.exit(1)
        print(f"(Mode n_envs=1 — tous les stages dans {tb_dir.relative_to(HERE)})")
        data = load_events(tb_dir)

        all_steps = sorted({s for tag in data for s, _ in data[tag]})
        total = all_steps[-1] if all_steps else 0

        if args.stages:
            boundaries = [0]
            for s in args.stages:
                boundaries.append(boundaries[-1] + s)
        else:
            # Découpe équitable par défaut
            chunk = total // n_stages
            boundaries = [i * chunk for i in range(n_stages + 1)]
            boundaries[-1] = total
            print(f"(Découpe auto : {[b for b in boundaries]} — "
                  f"passe --stages pour préciser)")

        for i, n in enumerate(stage_nums):
            lo, hi = boundaries[i], boundaries[i + 1]
            summarise(data, lo, hi, f"Stage {n}")

    print()


if __name__ == "__main__":
    main()
