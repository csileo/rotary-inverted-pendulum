"""Regenerate this repo's difficulty-level branches (2-full-no-models,
3-rl-ready, 4-diy-rl, demo) from `main`, without ever touching `main` or
the working tree.

Commits only ever land on `main` (level 1: everything, including models).
Each level branch is therefore not merged from `main` — it is fully
recomputed every run: `main`'s current flat file listing is filtered
(blocklist for level 2, explicit allowlist for levels 3/4/5), written into a
fresh git tree object, and committed as the new branch tip with two parents
(the branch's previous tip, if any, and `main`'s tip). That keeps pushing
the branch a fast-forward for anyone who already cloned/pulled it, and
avoids merge-conflict resolution entirely (a file added upstream inside an
already-excluded folder is simply filtered out again on every run, instead
of silently leaking in the way a real `git merge` would).

Level 5 also renames a couple of files on the way in (see REMAP below): its
README.md and requirements.txt live on `main` under different names
(`README.5-demo.md`, `requirements.5-demo.txt`) so they don't leak into
levels 2/3/4, which also pull from the same `RotaryInvertedPendulum-python/
src/rl/` directory and would otherwise get a demo-scoped requirements.txt
instead of a full one, or a demo README instead of `main`'s real one.

No dependencies beyond the standard library and a `git` on PATH.

Usage:
    python tools/sync_fork_branches.py [--dry-run]
    python tools/sync_fork_branches.py --levels 3 4
    python tools/sync_fork_branches.py --push

Content per level (see docs/rig_geometry_parameters.md and the RL docs for
what levels 3/4 are meant to teach), see also README.md's "Branches" table:
- 2-full-no-models: main minus models/ and logs/ (see README.md's "Branches"
  table — tools/ is NOT excluded here, it's just not a level-3/4 dependency).
- 3-rl-ready: RotaryInvertedPendulum-python/src/rl/ (minus sysid_wizard.py,
  models/, logs/) + the RL-paradigm docs + urdf/model.urdf (required
  at runtime by pendulum_geometry.py, even though the meshes are not).
- 4-diy-rl: sysid_params.json + sysid_runbook.md + rig_geometry_parameters.md
  only - no RL/sim code, no URDF, so the colleague has to write their own.
- demo: just enough to run inference against the real rig with the
  three reference models (working balance, partial balance, fails to
  swing up - run_policy.py's actual runtime deps only - NOT
  async_control.py/real_env.py, which are finetune_async.py-only), plus
  train_sac.py's simulation-only --eval path (pendulum_env.py,
  pendulum_geometry.py, reward.py, sysid_params.json, urdf/model.urdf -
  same "meshes not required" reasoning as level 3), the LowLevelServer
  firmware to flash, an ad-hoc README.md (flash + install + run commands)
  and a requirements.txt that adds gymnasium + mujoco on top of the
  inference-only set for that --eval path - not the generic
  RotaryInvertedPendulum-arduino/README.md (documents sketches this level
  doesn't even have) and not a training-sized requirements list. Also
  carries tools/pi_demo/ (run_demo.py + helpers): an unattended, pure-
  Python (Linux/macOS/Windows alike) launcher that tolerates any plug-in
  order/delay for power, the pendulum's 12V, and the Nano's USB cable, and
  skips reflashing the Nano when its firmware (checked via
  CMD_GET_FIRMWARE_VERSION) already matches.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_BRANCH = "main"

RL_DIR = "RotaryInvertedPendulum-python/src/rl"

ALWAYS_FILES = ["LICENSE"]

LEVEL2_EXCLUDE_PREFIXES = [
    f"{RL_DIR}/models/",
    f"{RL_DIR}/logs/",
    # demo-only sources, remapped for level 5 (see REMAP below) - the raw
    # ".5-demo" files must not leak into other levels unremapped.
    "README.5-demo.md",
    "requirements.5-demo.txt",
]

LEVEL3_EXCLUDE = [
    f"{RL_DIR}/sysid_wizard.py",
    f"{RL_DIR}/models/",
    f"{RL_DIR}/logs/",
]
LEVEL3_EXTRA_FILES = [
    "docs/domain_randomization.md",
    "docs/rl_transitions.md",
    "docs/quantisation.md",
    "docs/async_control_architecture.md",
    "docs/control_rate_selection.md",
    "docs/transport_delay.md",
    "urdf/model.urdf",
    "requirements.txt",
]

LEVEL4_FILES = [
    f"{RL_DIR}/sysid_params.json",
    "docs/sysid_runbook.md",
    "docs/rig_geometry_parameters.md",
]

# (source path on main, destination path in the branch tree) - identity for
# most entries, remapped for the ad-hoc README/requirements (see module
# docstring for why those two need different names on main).
LEVEL5_FILES = [
    (f"{RL_DIR}/run_policy.py", f"{RL_DIR}/run_policy.py"),
    (f"{RL_DIR}/frame_stack.py", f"{RL_DIR}/frame_stack.py"),
    (f"{RL_DIR}/lowlevel_client.py", f"{RL_DIR}/lowlevel_client.py"),
    (f"{RL_DIR}/train_sac.py", f"{RL_DIR}/train_sac.py"),
    (f"{RL_DIR}/pendulum_env.py", f"{RL_DIR}/pendulum_env.py"),
    (f"{RL_DIR}/pendulum_geometry.py", f"{RL_DIR}/pendulum_geometry.py"),
    (f"{RL_DIR}/reward.py", f"{RL_DIR}/reward.py"),
    (f"{RL_DIR}/sysid_params.json", f"{RL_DIR}/sysid_params.json"),
    ("urdf/model.urdf", "urdf/model.urdf"),
    (
        f"{RL_DIR}/models/policy_working_balance.zip",
        f"{RL_DIR}/models/policy_working_balance.zip",
    ),
    (
        f"{RL_DIR}/models/policy_fails_no_swingup.zip",
        f"{RL_DIR}/models/policy_fails_no_swingup.zip",
    ),
    (
        f"{RL_DIR}/models/policy_partial_balance.zip",
        f"{RL_DIR}/models/policy_partial_balance.zip",
    ),
    (
        "RotaryInvertedPendulum-arduino/LowLevelServer/LowLevelServer.ino",
        "RotaryInvertedPendulum-arduino/LowLevelServer/LowLevelServer.ino",
    ),
    (
        "RotaryInvertedPendulum-arduino/LowLevelServer/StepperUtils.h",
        "RotaryInvertedPendulum-arduino/LowLevelServer/StepperUtils.h",
    ),
    (
        "RotaryInvertedPendulum-arduino/LowLevelServer/firmware_version.h",
        "RotaryInvertedPendulum-arduino/LowLevelServer/firmware_version.h",
    ),
    (
        "RotaryInvertedPendulum-arduino/LowLevelServer/gen_firmware_version.py",
        "RotaryInvertedPendulum-arduino/LowLevelServer/gen_firmware_version.py",
    ),
    ("tools/pi_demo/pi_demo_common.py", "tools/pi_demo/pi_demo_common.py"),
    ("tools/pi_demo/flash_if_needed.py", "tools/pi_demo/flash_if_needed.py"),
    ("tools/pi_demo/check_motor_power.py", "tools/pi_demo/check_motor_power.py"),
    ("tools/pi_demo/run_demo.py", "tools/pi_demo/run_demo.py"),
    ("tools/pi_demo/README.md", "tools/pi_demo/README.md"),
    ("README.5-demo.md", "README.md"),
    ("requirements.5-demo.txt", "requirements.txt"),
]


def _level2_keep(path: str) -> bool:
    return not any(path.startswith(p) for p in LEVEL2_EXCLUDE_PREFIXES)


def _level3_keep(path: str) -> bool:
    if path in LEVEL3_EXTRA_FILES:
        return True
    if not path.startswith(f"{RL_DIR}/"):
        return False
    return not any(
        path.startswith(p) if p.endswith("/") else path == p
        for p in LEVEL3_EXCLUDE
    )


def _allowlist(files: list[str]):
    """Plain allowlist, no renaming: source path == destination path."""
    pairs = [(f, f) for f in files]
    return _allowlist_from_pairs(pairs)


def _allowlist_from_pairs(pairs: list[tuple[str, str]]):
    """Allowlist with (source path, destination path) remapping."""
    remap = dict(pairs)
    keep = lambda path: path in remap
    expected = [src for src, _ in pairs]
    return keep, remap, expected


# level number -> (branch name, predicate, remap or None, expected sources)
LEVELS = {
    2: ("2-full-no-models", _level2_keep, None, None),
    3: ("3-rl-ready", _level3_keep, None, LEVEL3_EXTRA_FILES),
    4: ("4-diy-rl", *_allowlist(LEVEL4_FILES)),
    5: ("demo", *_allowlist_from_pairs(LEVEL5_FILES)),
}


def run_git(args: list[str], env: dict | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, env=env, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--levels", type=int, nargs="+", choices=sorted(LEVELS), default=sorted(LEVELS)
    )
    parser.add_argument(
        "--push", action="store_true", help="Push each updated branch to origin."
    )
    args = parser.parse_args()

    main_sha = run_git(["rev-parse", SOURCE_BRANCH]).strip()
    main_subject = run_git(["log", "-1", "--format=%s", main_sha]).strip()
    print(f"Source: {SOURCE_BRANCH} @ {main_sha[:10]} ({main_subject})\n")

    all_lines = run_git(["ls-tree", "-r", "--full-tree", main_sha]).splitlines()

    for level in args.levels:
        branch, keep, remap, expected = LEVELS[level]
        filtered_srcs: set[str] = set()
        filtered_lines = []
        for line in all_lines:
            mode_type_sha, src_path = line.split("\t", 1)
            if not (keep(src_path) or src_path in ALWAYS_FILES):
                continue
            filtered_srcs.add(src_path)
            dst_path = remap.get(src_path, src_path) if remap else src_path
            filtered_lines.append(f"{mode_type_sha}\t{dst_path}")

        if expected:
            for f in expected:
                if f not in filtered_srcs:
                    print(f"  WARNING {branch}: expected file not found in {SOURCE_BRANCH}: {f}")

        print(f"{branch}: {len(filtered_lines)} files")

        if args.dry_run:
            for line in sorted(filtered_lines, key=lambda l: l.split("\t", 1)[1]):
                print(f"    {line.split(chr(9), 1)[1]}")
            continue

        fd, tmp_index = tempfile.mkstemp(prefix="sync-index-", dir=REPO / ".git")
        os.close(fd)
        os.remove(tmp_index)  # git wants the path free, not a pre-existing empty file
        env = {**os.environ, "GIT_INDEX_FILE": tmp_index}
        try:
            run_git(["read-tree", "--empty"], env=env)
            # --add is required: without it, update-index silently drops
            # every path as "ignoring new file" since the fresh index has
            # nothing to "update" yet. Input must be raw bytes, not
            # text=True: on Windows, text-mode stdin translates '\n' to
            # '\r\n', which corrupts the tab-delimited ls-tree format and
            # makes update-index silently ignore every path too (both bit
            # us once already - produced empty-tree commits with no error).
            subprocess.run(
                ["git", "update-index", "--add", "--index-info"],
                cwd=REPO,
                env=env,
                input=("\n".join(filtered_lines) + "\n").encode("utf-8"),
                check=True,
            )
            new_tree = run_git(["write-tree"], env=env).strip()
        finally:
            if os.path.exists(tmp_index):
                os.remove(tmp_index)

        prev = run_git(["rev-parse", "--verify", f"refs/heads/{branch}"], check=False).strip()
        if prev:
            prev_tree = run_git(["rev-parse", f"{prev}^{{tree}}"]).strip()
            if prev_tree == new_tree:
                print(f"    no changes since last sync ({prev[:10]})")
                continue
            parents = ["-p", prev, "-p", main_sha]
        else:
            parents = ["-p", main_sha]

        message = f"Sync {branch} from {SOURCE_BRANCH} ({main_sha[:10]})"
        new_commit = run_git(["commit-tree", new_tree, *parents, "-m", message]).strip()
        run_git(["update-ref", f"refs/heads/{branch}", new_commit])
        print(f"    {branch} -> {new_commit[:10]}")

        if args.push:
            run_git(["push", "origin", branch])
            print(f"    pushed to origin/{branch}")

    print()


if __name__ == "__main__":
    main()
