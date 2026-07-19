"""DEPRECATED — use `finetune_async.py` instead.

This script used to call SB3's `model.learn()` against the real-device env.
That worked, but `learn()` runs `collect_rollouts` and `train()` sequentially
in one thread, so gradient updates eat into the per-step time budget. With
`--gradient-steps 4` the actual control rate dropped to ~35 Hz against a
configured 100 Hz, and the policy silently learned the wrong dynamics.
See `docs/rl_transitions.md` and `RL_PLAN.md` Phase 4.5 for the diagnosis.

`finetune_async.py` is the proper replacement: it owns a background thread
that drives the rig at strict 100 Hz independent of how slow the learner
is. All flags map across (with `--resume-buffer` added for cross-session
real-data persistence).

This shim forwards your CLI args to `finetune_async.main()` and prints a
deprecation notice so existing scripts and shell history keep working.
"""

from __future__ import annotations

import sys

from finetune_async import main as _async_main


_DEPRECATION_NOTICE = """\
══════════════════════════════════════════════════════════════════════════════
DEPRECATION: finetune_real.py is now a thin shim. Forwarding to
finetune_async.py, which holds the configured control rate strictly.

Update your scripts to call finetune_async.py directly. CLI flags are
identical except --resume-buffer is new (carry real-robot transitions
across sessions). See docs/rl_transitions.md for context.
══════════════════════════════════════════════════════════════════════════════
"""


def main(argv: list[str] | None = None) -> int:
    print(_DEPRECATION_NOTICE, file=sys.stderr)
    return _async_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
