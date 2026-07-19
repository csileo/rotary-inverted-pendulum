"""Résume la progression d'un run finetune_async.py à partir de ses logs TensorBoard.

finetune_async.py écrase timing.csv à chaque lancement (open(..., "w")), donc
si un run a été relancé plusieurs fois (crash, arrêt manuel) sous le même
--run-name, seul le dernier lancement survit dans timing.csv. Les sous-dossiers
tb/async_N (un par lancement du process, SB3 les numérote automatiquement)
restent en revanche tous sur disque et permettent de reconstituer l'historique
complet, lancement par lancement.

Usage:
    python analyze_finetune_tb.py <chemin_vers_run_dir>
    python analyze_finetune_tb.py <chemin_vers_run_dir> --min-episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path


METRICS = [
    ("rollout/ep_reward", "ep_reward   "),
    ("rollout/ep_rew_mean", "ep_rew_mean "),
    ("rollout/ep_length", "ep_length   "),
    ("train/actor_loss", "actor_loss  "),
    ("train/critic_loss", "critic_loss "),
    ("train/ent_coef", "ent_coef    "),
]


def load_events(tb_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return {t: [(e.step, e.wall_time, e.value) for e in ea.Scalars(t)] for t in tags}


def summarise(data: dict, label: str, min_episodes: int) -> bool:
    ep_pts = data.get("rollout/ep_reward", [])
    n_episodes = len(ep_pts)
    if n_episodes < min_episodes:
        return False

    t0 = min(w for _, w, _ in (p for pts in data.values() for p in pts)) if data else 0.0
    t1 = max(w for _, w, _ in (p for pts in data.values() for p in pts)) if data else 0.0
    duration_min = (t1 - t0) / 60.0

    print(f"\n{'=' * 60}")
    print(f"  {label}  —  {n_episodes} épisodes, ~{duration_min:.1f} min")
    print(f"{'=' * 60}")

    for tag, name in METRICS:
        pts = data.get(tag, [])
        if not pts:
            print(f"  {name}: n/a")
            continue
        vals = [v for _, _, v in pts]
        first_v, last_v = vals[0], vals[-1]
        arrow = "v" if last_v < first_v else "^"
        # Moyenne glissante début/fin pour amortir le bruit épisode-à-épisode.
        k = max(1, len(vals) // 5)
        mean_first = sum(vals[:k]) / k
        mean_last = sum(vals[-k:]) / k
        print(f"  {name}: {first_v:+.3f} -> {last_v:+.3f}  {arrow}  "
              f"[début~{mean_first:+.3f}  fin~{mean_last:+.3f}]  "
              f"(min {min(vals):.3f}  max {max(vals):.3f}  n={len(vals)})")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="dossier du run (contient tb/async_*)")
    p.add_argument("--min-episodes", type=int, default=3,
                   help="ignore les lancements avortés avec moins d'épisodes que ça")
    args = p.parse_args()

    tb_root = args.run_dir / "tb"
    if not tb_root.exists():
        print(f"Pas de dossier tb/ dans {args.run_dir}")
        raise SystemExit(1)

    sub_dirs = sorted(
        (d for d in tb_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )

    print(f"Run : {args.run_dir}")
    print(f"{len(sub_dirs)} lancement(s) trouvé(s) sous tb/")

    kept = 0
    for d in sub_dirs:
        data = load_events(d)
        if summarise(data, d.name, args.min_episodes):
            kept += 1
    if kept == 0:
        print("\nAucun lancement n'a atteint --min-episodes ; tous ont crashé tôt.")
    print()


if __name__ == "__main__":
    main()
