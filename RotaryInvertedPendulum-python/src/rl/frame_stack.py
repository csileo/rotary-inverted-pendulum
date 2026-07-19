"""Frame stacking for the pendulum observation.

`motor_vel`/`pen_vel` in the observation are a noisy, laggy estimate (on the
real rig, an 8ms-window linear regression over the AS5600 encoder — see
`real_env.py`'s `REST_THRESHOLD_RAD_S` comment). Stacking the last few raw
observations lets the policy reconstruct its own velocity/derivative estimate
from position history instead of depending on that smoothed-but-lagged value.

Used identically by `pendulum_env.py` (sim), `real_env.py` (rig fine-tuning),
and `run_policy.py` (standalone hardware runner) so a checkpoint trained in
one place runs unmodified in the others. `FrameStacker` only ever sees the
opaque per-step observation array — it knows nothing about the pendulum.

`reset()`/`push()` mutate internal state and must each be called exactly
once per physical env transition (one call per `reset()`/`step()` in the
env, one call per control tick in `run_policy.py`). Calling either an extra
time desyncs the stack from what actually happened physically.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class FrameStacker:
    """Rolling window of the last `n_frames` observations, concatenated
    oldest -> newest.

    With `n_frames=1`, `reset()`/`push()` degenerate to the identity
    transform on a single frame — this is what makes frame_stack=1 (the
    default everywhere) bit-for-bit backward compatible with checkpoints
    trained before frame stacking existed.
    """

    def __init__(self, n_frames: int, frame_dim: int):
        self.n_frames = int(n_frames)
        self.frame_dim = int(frame_dim)
        self._buf: deque[np.ndarray] = deque(maxlen=self.n_frames)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Fill the buffer with `n_frames` copies of `frame` (no fabricated
        history at episode start) and return the stacked observation."""
        self._buf.clear()
        for _ in range(self.n_frames):
            self._buf.append(frame)
        return self._stacked()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Append `frame`, evicting the oldest if the buffer is full, and
        return the stacked observation."""
        self._buf.append(frame)
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._buf)).astype(np.float32)
