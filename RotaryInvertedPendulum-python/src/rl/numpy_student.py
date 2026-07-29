"""Pure-NumPy inference for a distilled student MLP — no PyTorch import.

`distill.py` trains and exports a `StudentMLP` (PyTorch) as `student.pt`.
`export_weights_numpy.py` converts that checkpoint (on a machine that
already has torch, e.g. the dev PC) into a plain `.npz` of the raw weight
arrays. This module then re-implements the exact same forward pass
(`numpy_forward`, shared with `distill.py`'s own parity check) so a
runtime that only ever needs to *run* the student — like the Raspberry Pi
demo loop — never has to import torch/stable-baselines3 at all.

Usage:
    weights, meta = load_numpy_student("student_numpy.npz")
    predict = make_numpy_predict(weights)
    action, _ = predict(obs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_WEIGHT_KEYS = ("W1", "B1", "W2", "B2", "W3", "B3")


def numpy_forward(weights: dict, x: np.ndarray) -> np.ndarray:
    """Forward pass mirroring the PyTorch StudentMLP: ReLU, ReLU, tanh."""
    h1 = np.maximum(0.0, x @ weights["W1"].T + weights["B1"])
    h2 = np.maximum(0.0, h1 @ weights["W2"].T + weights["B2"])
    return np.tanh(h2 @ weights["W3"].T + weights["B3"])


def load_numpy_student(npz_path: str | Path) -> tuple[dict, dict]:
    """Load a `.npz` produced by `export_weights_numpy.py`.

    Returns (weights, meta) where weights holds W1/B1/W2/B2/W3/B3 as
    float32 arrays and meta holds hidden/obs_dim/act_dim/val_mse.
    """
    data = np.load(npz_path)
    weights = {k: np.asarray(data[k], dtype=np.float32) for k in _WEIGHT_KEYS}
    meta = {
        "hidden": int(data["hidden"]),
        "obs_dim": int(data["obs_dim"]),
        "act_dim": int(data["act_dim"]),
        "val_mse": float(data["val_mse"]),
    }
    return weights, meta


def make_numpy_predict(weights: dict):
    """Return a function with SB3-style `predict(obs, deterministic=True)`."""

    def predict(obs, deterministic: bool = True):  # noqa: ARG001 (deterministic always true)
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        a = numpy_forward(weights, arr)
        return (a[0] if a.shape[0] == 1 else a), None

    return predict
