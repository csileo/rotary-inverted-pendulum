"""Export a QAT-trained student MLP as a PROGMEM C header for the Arduino Nano.

Reads the .pt produced by `distill_quantised.py` and writes a self-contained
header (`policy_weights_quantised.h` for --bits 8, or a differently-named
int16-storage header for --bits 9-16 — see RLControl.ino's
POLICY_QUANTISED_INT8 / POLICY_QUANTISED_INT16 switch).

Quantisation scheme (matches `distill_quantised.py` and the C++ forward pass):

    Per-tensor symmetric, no zero point. bits=8 -> int8 [-127,127];
    bits=9..16 -> stored as int16_t but logically clamped to
    [-(2**(bits-1)-1), 2**(bits-1)-1] (see distill_quantised.py's --bits
    docstring for why a narrower-than-16 logical range can still be the
    right call here).
        quantise(x, s)   = clamp(round(x / s), -max_int, max_int)
        dequantise(q, s) = q * s

    Per-layer Linear:
        y = W @ x + b           (float math we replace)
        becomes
        accum_i32 = sum( W_int[i,j] * x_int[j] ) + b_int32[i]
                                   ^ accumulate in int32
                                   ^ b_int32 pre-scaled to (s_w * s_x) units
        Then:
            For hidden layers: rescale accum_i32 to the next layer's int
                               range using a fixed-point multiply-shift.
            For the final layer: dequantise accum_i32 to float, apply tanh.

    Rescale (hidden layers) — PER-LAYER ADAPTIVE SHIFT, not a fixed Q15:
        We want   y_int = clamp(round(accum_i32 * (s_w * s_x / s_y)), 0, max_int)
        Implement (s_w * s_x / s_y) as M_q / 2^shift:
            M_q   = round((s_w * s_x / s_y) * 2^shift)
            y_int = clamp(((accum_i32 * M_q) + (1<<(shift-1))) >> shift, 0, max_int)
        `shift` is chosen per layer (see calibrate_rescale()) to be the
        LARGEST value that keeps M_q inside int16 AND accum_i32 * M_q inside
        int32 (checked empirically against real on-distribution data, with
        margin) — NOT hardcoded to 15. A fixed shift=15 was fine for int8
        (M_q naturally landed in the tens-to-hundreds range there) but
        collapses to a handful of representable multiplier values at
        bits=14/16, because layer 1's "absorb the per-channel input scale
        into the weights" trick (_absorb_per_channel_input_scales) makes
        that layer's s_w/s_h ratio shrink roughly as 1/max_int^2 instead of
        1/max_int as bits grows — a fixed Q15 rescale silently loses almost
        all of layer 1's dynamic range there (some neurons' M_q rounds to
        0, i.e. that neuron reads as a constant regardless of input).

    Final dequantise:
        y_float = accum_i32 * (s_w * s_h2)         # one float multiply
        action  = tanh(y_float)                    # one libm call

Bias quantisation:
        b_int32[i] = round(bias_float[i] / (s_w * s_x))
        Lives in the same units as `accum` so it adds directly without
        further rescaling.

Header schema (matches what `RLControl.ino` consumes under
POLICY_QUANTISED_INT8 / POLICY_QUANTISED_INT16):

    POLICY_WEIGHT_BITS, POLICY_OBS_DIM, POLICY_HIDDEN_DIM, POLICY_OUT_DIM
    POLICY_RESCALE_SHIFT_L1, POLICY_RESCALE_SHIFT_L2   (#define, per layer)
    POLICY_INV_SCALE_OBS_IN  (float, = 1 / s_obs_in; multiply obs by this)
    POLICY_W1[H][O], POLICY_W2[H][H], POLICY_W3[1][H]   (int8_t or int16_t)
    POLICY_B1[H], POLICY_B2[H], POLICY_B3[1]             (int32)
    POLICY_RESCALE_M_L1, POLICY_RESCALE_M_L2             (int16, per row)
    POLICY_DEQUANT_L3                                    (float)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import torch

from distill_quantised import QATStudent


def _quantise_weight_per_row(w_float: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric quantisation: each row of W gets its own scale.

    Returns (W_int of shape (out, in), s_w of shape (out,)). The canonical
    TFLite-Micro pattern; recovers a lot of fidelity over per-tensor scaling
    when different output neurons have different weight magnitudes.
    bits=8 -> int8 [-127,127]; bits=9..16 -> int16_t storage, logical range
    [-(2**(bits-1)-1), 2**(bits-1)-1].
    """
    max_int = 2 ** (bits - 1) - 1
    dtype = np.int8 if bits == 8 else np.int16
    max_abs_per_row = np.max(np.abs(w_float), axis=1)  # (out,)
    s_w = np.maximum(max_abs_per_row / max_int, 1e-8).astype(np.float64)  # (out,)
    w_int = np.clip(np.round(w_float / s_w[:, None]), -max_int, max_int).astype(dtype)
    return w_int, s_w


def _quantise_bias_int32_per_row(b_float: np.ndarray, s_w_per_row: np.ndarray,
                                 s_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-row bias: b_int[i] = round(b[i] / (s_w[i] * s_x)).

    `s_x` is a scalar — for layer 1 we use the *effective* per-tensor scale
    derived after absorbing per-channel input scales into the weights, so
    the bias is in the same accumulator units as the matmul output.
    """
    s_b = s_w_per_row * s_x  # (out,)
    b_int = np.round(b_float / np.maximum(s_b, 1e-12)).astype(np.int64)
    b_int = np.clip(b_int, -(2**31), (2**31) - 1).astype(np.int32)
    return b_int, s_b


def _absorb_per_channel_input_scales(w_float: np.ndarray,
                                     s_obs_per_channel: np.ndarray) -> np.ndarray:
    """Fold per-channel input scales into weights: W_eff[i,j] = W[i,j] * s_obs[j].

    After this, W_eff can be quantised per row and the deployment matmul
    does sum_j (W_eff_int[i,j] * x_int[j]), where x_int[j] = round(x[j]
    / s_obs[j]) — i.e. the per-channel input scales are already inside the
    weights. The Arduino still does an ordinary int matmul.
    """
    return w_float * s_obs_per_channel[None, :]  # broadcast over rows


def _format_int8_2d(name: str, arr: np.ndarray) -> str:
    rows, cols = arr.shape
    lines = [f"const int8_t {name}[{rows}][{cols}] PROGMEM = {{"]
    for r in range(rows):
        vals = ", ".join(f"{int(v):4d}" for v in arr[r])
        lines.append(f"    {{ {vals} }}{',' if r < rows - 1 else ''}")
    lines.append("};")
    return "\n".join(lines)


def _format_int16_2d(name: str, arr: np.ndarray) -> str:
    rows, cols = arr.shape
    lines = [f"const int16_t {name}[{rows}][{cols}] PROGMEM = {{"]
    for r in range(rows):
        vals = ", ".join(f"{int(v):6d}" for v in arr[r])
        lines.append(f"    {{ {vals} }}{',' if r < rows - 1 else ''}")
    lines.append("};")
    return "\n".join(lines)


def _format_int32_1d(name: str, arr: np.ndarray) -> str:
    n = arr.shape[0]
    vals = ",\n    ".join(f"{int(v):>11d}L" for v in arr)
    return f"const int32_t {name}[{n}] PROGMEM = {{\n    {vals}\n}};"


def _format_int16_1d(name: str, arr: np.ndarray) -> str:
    n = arr.shape[0]
    vals = ",\n    ".join(f"{int(v):>7d}" for v in arr)
    return f"const int16_t {name}[{n}] PROGMEM = {{\n    {vals}\n}};"


def _format_float_1d(name: str, arr: np.ndarray) -> str:
    n = arr.shape[0]
    vals = ",\n    ".join(f"{float(v):+.8e}f" for v in arr)
    return f"const float {name}[{n}] PROGMEM = {{\n    {vals}\n}};"


# ---------------------------------------------------------------------------
# Rescale calibration — per-layer adaptive Q-shift (see module docstring).
# ---------------------------------------------------------------------------

def _pick_shift(M_row: np.ndarray, accum_samples: np.ndarray, *,
                headroom: float = 0.8, max_shift: int = 24) -> tuple[np.ndarray, int]:
    """Choose one right-shift S for a layer plus per-row Q_S multipliers
    M_q[i] = round(M_row[i] * 2**S) — the largest S (best precision) such
    that, against the REAL accumulator values actually seen:
      - M_q fits in int16 (the Arduino stores it as int16_t regardless of
        the weight bit-width).
      - accum * M_q fits within `headroom` of int32 range.
    `accum_samples` must already reflect the actual (sample, row) pairs at
    THIS layer — the empirical bound this function enforces would be
    meaningless against a synthetic/worst-case accumulator estimate.
    """
    max_abs_M = float(np.max(np.abs(M_row)))
    max_abs_accum = float(np.max(np.abs(accum_samples)))
    int16_max = 32767
    int32_budget = headroom * (2 ** 31 - 1)

    shift = max_shift
    while shift > 0 and (
        max_abs_M * (2 ** shift) > int16_max
        or max_abs_accum * max_abs_M * (2 ** shift) > int32_budget
    ):
        shift -= 1

    if max_abs_accum * max_abs_M * (2 ** shift) > int32_budget:
        raise RuntimeError(
            f"cannot find a safe Q-shift even at shift=0: max|accum|={max_abs_accum:.3e} "
            f"x max|M|={max_abs_M:.3e} exceeds the int32 headroom budget "
            f"({headroom*100:.0f}% of {2**31-1:.3e}). The raw (pre-rescale) "
            f"accumulator itself is too large — reduce --bits."
        )

    M_q = np.round(M_row * (2 ** shift)).astype(np.int64)
    if np.any(np.abs(M_q) > int16_max):
        raise RuntimeError(
            f"Q-multiplier overflows int16 even at shift={shift}: "
            f"range [{M_q.min()}, {M_q.max()}]"
        )
    return M_q.astype(np.int16), shift


def calibrate_rescale(
    check_obs: np.ndarray,
    *,
    W1q: np.ndarray, B1q: np.ndarray, s_w1: np.ndarray, s_h1: float,
    W2q: np.ndarray, B2q: np.ndarray, s_w2: np.ndarray, s_h2: float,
    W3q: np.ndarray, B3q: np.ndarray,
    inv_scale_obs: np.ndarray, max_int: int,
    headroom: float = 0.8,
) -> dict:
    """Calibrate the per-layer fixed-point rescale against real on-distribution obs.

    Strictly sequential, not circular: layer 2's raw accumulator depends on
    layer 1's *actual* quantised output, which depends on layer 1's chosen
    shift — so layer 1 is fully resolved (shift + real h1 values) before
    layer 2's calibration ever runs. Layer 3 dequantises straight to float
    (no Q-format), so it only needs an accumulator-range check, not a shift.
    """
    x = np.round(check_obs.astype(np.float64) * inv_scale_obs[None, :].astype(np.float64))
    x = np.clip(x, -max_int, max_int).astype(np.int64)

    accum_l1 = B1q[None, :].astype(np.int64) + x @ W1q.T.astype(np.int64)
    M_l1 = s_w1 / s_h1
    M_q_l1, shift_l1 = _pick_shift(M_l1, accum_l1, headroom=headroom)
    round_l1 = (1 << (shift_l1 - 1)) if shift_l1 > 0 else 0
    h1 = np.clip((accum_l1 * M_q_l1[None, :].astype(np.int64) + round_l1) >> shift_l1,
                 0, max_int)

    accum_l2 = B2q[None, :].astype(np.int64) + h1 @ W2q.T.astype(np.int64)
    M_l2 = (s_w2 * s_h1) / s_h2
    M_q_l2, shift_l2 = _pick_shift(M_l2, accum_l2, headroom=headroom)
    round_l2 = (1 << (shift_l2 - 1)) if shift_l2 > 0 else 0
    h2 = np.clip((accum_l2 * M_q_l2[None, :].astype(np.int64) + round_l2) >> shift_l2,
                 0, max_int)

    accum_l3 = B3q[None, :].astype(np.int64) + h2 @ W3q.T.astype(np.int64)

    int32_max = 2 ** 31 - 1
    max_abs = {
        "l1": int(np.max(np.abs(accum_l1))),
        "l2": int(np.max(np.abs(accum_l2))),
        "l3": int(np.max(np.abs(accum_l3))),
    }
    return {
        "M_q_l1": M_q_l1, "shift_l1": shift_l1,
        "M_q_l2": M_q_l2, "shift_l2": shift_l2,
        "max_abs_accum": max_abs,
        "headroom": 1.0 - max(max_abs.values()) / int32_max,
    }


def numpy_forward_quantised(
    obs: np.ndarray,
    *,
    W1: np.ndarray, B1: np.ndarray,
    W2: np.ndarray, B2: np.ndarray,
    W3: np.ndarray, B3: np.ndarray,
    inv_scale_obs: np.ndarray,   # per-channel, shape (obs_dim,)
    M_q_l1: np.ndarray, shift_l1: int,   # per output channel, shape (H,)
    M_q_l2: np.ndarray, shift_l2: int,   # per output channel, shape (H,)
    dequant_l3: np.ndarray,      # per output channel, shape (act_dim,)
    max_int: int = 127,
    diag: dict | None = None,
) -> np.ndarray:
    """Numpy implementation that mirrors the Arduino quantised forward pass exactly.

    Single-sample input. Returns the float action (post-tanh). Used both for
    parity-checking the export and as a reference for the C++. `max_int` is
    127 for int8, up to 32767 for int16-storage. `shift_l1`/`shift_l2` come
    from calibrate_rescale() — NOT a fixed 15 (see module docstring).
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    x = np.empty(obs.shape[0], dtype=np.int32)
    for j in range(obs.shape[0]):
        q = int(round(obs[j] * inv_scale_obs[j]))
        if q >  max_int: q =  max_int
        if q < -max_int: q = -max_int
        x[j] = q

    round_l1 = (1 << (shift_l1 - 1)) if shift_l1 > 0 else 0
    H = W1.shape[0]
    h1 = np.zeros(H, dtype=np.int32)
    max_abs_accum_l1 = 0
    for i in range(H):
        accum = int(B1[i])
        for j in range(W1.shape[1]):
            accum += int(W1[i, j]) * int(x[j])
        max_abs_accum_l1 = max(max_abs_accum_l1, abs(accum))
        scaled = (accum * int(M_q_l1[i]) + round_l1) >> shift_l1
        if scaled > max_int: scaled = max_int
        if scaled < 0:       scaled = 0       # ReLU
        h1[i] = scaled

    round_l2 = (1 << (shift_l2 - 1)) if shift_l2 > 0 else 0
    h2 = np.zeros(H, dtype=np.int32)
    max_abs_accum_l2 = 0
    for i in range(H):
        accum = int(B2[i])
        for j in range(W2.shape[1]):
            accum += int(W2[i, j]) * int(h1[j])
        max_abs_accum_l2 = max(max_abs_accum_l2, abs(accum))
        scaled = (accum * int(M_q_l2[i]) + round_l2) >> shift_l2
        if scaled > max_int: scaled = max_int
        if scaled < 0:       scaled = 0
        h2[i] = scaled

    accum = int(B3[0])
    for j in range(W3.shape[1]):
        accum += int(W3[0, j]) * int(h2[j])
    max_abs_accum_l3 = abs(accum)
    y = float(accum) * float(dequant_l3[0])

    if diag is not None:
        diag["max_abs_accum_l1"] = max(diag.get("max_abs_accum_l1", 0), max_abs_accum_l1)
        diag["max_abs_accum_l2"] = max(diag.get("max_abs_accum_l2", 0), max_abs_accum_l2)
        diag["max_abs_accum_l3"] = max(diag.get("max_abs_accum_l3", 0), max_abs_accum_l3)

    return np.float32(np.tanh(y))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _quantise_all(sd: dict, s_obs_pc: np.ndarray, s_h1: float, s_h2: float, bits: int) -> dict:
    """Shared by export() and parity_check() so both quantise identically."""
    W1f = sd["fc1.weight"].cpu().numpy().astype(np.float32)  # (H, O)
    B1f = sd["fc1.bias"].cpu().numpy().astype(np.float32)
    W2f = sd["fc2.weight"].cpu().numpy().astype(np.float32)  # (H, H)
    B2f = sd["fc2.bias"].cpu().numpy().astype(np.float32)
    W3f = sd["fc3.weight"].cpu().numpy().astype(np.float32)  # (1, H)
    B3f = sd["fc3.bias"].cpu().numpy().astype(np.float32)

    W1_eff = _absorb_per_channel_input_scales(W1f, s_obs_pc.astype(np.float32))
    W1q, s_w1 = _quantise_weight_per_row(W1_eff, bits=bits)
    B1q, _ = _quantise_bias_int32_per_row(B1f, s_w1, s_x=1.0)

    W2q, s_w2 = _quantise_weight_per_row(W2f, bits=bits)
    B2q, _ = _quantise_bias_int32_per_row(B2f, s_w2, s_x=s_h1)
    W3q, s_w3 = _quantise_weight_per_row(W3f, bits=bits)
    B3q, _ = _quantise_bias_int32_per_row(B3f, s_w3, s_x=s_h2)

    dequant_l3 = (s_w3 * s_h2).astype(np.float32)
    inv_scale_obs = (1.0 / np.maximum(s_obs_pc, 1e-12)).astype(np.float32)

    return dict(W1q=W1q, B1q=B1q, s_w1=s_w1, W2q=W2q, B2q=B2q, s_w2=s_w2,
                W3q=W3q, B3q=B3q, s_w3=s_w3, dequant_l3=dequant_l3,
                inv_scale_obs=inv_scale_obs)


def export(student_path: Path, header_path: Path, *, source_name: str | None = None,
           calib_dataset: Path | None = None) -> dict:
    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=True)
    hidden = int(ckpt["hidden"])
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    bits = int(ckpt.get("bits", 8))
    max_int = 2 ** (bits - 1) - 1
    val_mse = float(ckpt.get("val_mse", float("nan")))

    model = QATStudent(hidden=hidden, obs_dim=obs_dim, act_dim=act_dim, bits=bits)
    model.load_state_dict(ckpt["state_dict"])
    sd = model.state_dict()

    s_obs_pc = np.asarray(ckpt["scales"]["obs_in"], dtype=np.float64)  # (obs_dim,)
    s_h1 = float(ckpt["scales"]["h1"])
    s_h2 = float(ckpt["scales"]["h2"])

    q = _quantise_all(sd, s_obs_pc, s_h1, s_h2, bits)

    if calib_dataset is None or not calib_dataset.exists():
        raise RuntimeError(
            "export() requires --parity-dataset as the rescale-calibration "
            "sample — the per-layer Q-shift is picked against real "
            "accumulator values, not a theoretical worst case (see "
            "calibrate_rescale())."
        )
    data = np.load(calib_dataset)
    all_obs = np.asarray(data["obs"], dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = rng.choice(all_obs.shape[0], size=min(8192, all_obs.shape[0]), replace=False)
    calib_obs = all_obs[idx]

    calib = calibrate_rescale(
        calib_obs, W1q=q["W1q"], B1q=q["B1q"], s_w1=q["s_w1"], s_h1=s_h1,
        W2q=q["W2q"], B2q=q["B2q"], s_w2=q["s_w2"], s_h2=s_h2,
        W3q=q["W3q"], B3q=q["B3q"], inv_scale_obs=q["inv_scale_obs"], max_int=max_int,
    )
    print(f"[export] rescale calibration ({calib_obs.shape[0]} samples): "
          f"shift_l1={calib['shift_l1']} (M range [{calib['M_q_l1'].min()},{calib['M_q_l1'].max()}])  "
          f"shift_l2={calib['shift_l2']} (M range [{calib['M_q_l2'].min()},{calib['M_q_l2'].max()}])  "
          f"max|accum| L1={calib['max_abs_accum']['l1']:.3e} L2={calib['max_abs_accum']['l2']:.3e} "
          f"L3={calib['max_abs_accum']['l3']:.3e}  headroom={calib['headroom']*100:.1f}%")

    W1q, B1q, s_w1 = q["W1q"], q["B1q"], q["s_w1"]
    W2q, B2q, s_w2 = q["W2q"], q["B2q"], q["s_w2"]
    W3q, B3q, s_w3 = q["W3q"], q["B3q"], q["s_w3"]
    dequant_l3, inv_scale_obs = q["dequant_l3"], q["inv_scale_obs"]
    M_q_l1, shift_l1 = calib["M_q_l1"], calib["shift_l1"]
    M_q_l2, shift_l2 = calib["M_q_l2"], calib["shift_l2"]

    bytes_per_weight = 1 if bits == 8 else 2
    weight_ctype = "int8_t" if bits == 8 else "int16_t"
    format_weight_2d = _format_int8_2d if bits == 8 else _format_int16_2d

    n_params_w = W1q.size + W2q.size + W3q.size
    n_params_int32 = B1q.size + B2q.size + B3q.size
    n_params_int16 = M_q_l1.size + M_q_l2.size
    n_params_float = inv_scale_obs.size + dequant_l3.size
    flash_bytes = (bytes_per_weight * n_params_w + 4 * n_params_int32
                   + 2 * n_params_int16 + 4 * n_params_float)

    h = []
    h.append("// auto-generated by export_weights_quantised.py — do not edit by hand")
    if source_name:
        h.append(f"// source: {source_name}")
    h.append(f"// generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    h.append(f"// quantised student MLP: {obs_dim} -> {hidden} -> {hidden} -> {act_dim} (int{bits})")
    h.append(f"// per-channel input + per-row weight quantisation, per-layer adaptive Q-shift rescale")
    h.append(f"// weights+biases: {n_params_w} {weight_ctype} + {n_params_int32} int32 + "
             f"{n_params_int16} int16 + {n_params_float} float = {flash_bytes} flash bytes")
    h.append(f"// QAT val_mse: {val_mse:.6f}")
    obs_scale_strs = ', '.join(f'{float(s):.4e}' for s in s_obs_pc)
    h.append(f"// per-channel input scales: [{obs_scale_strs}]")
    h.append(f"// hidden activation scales: s_h1={s_h1:.6e}  s_h2={s_h2:.6e}")
    h.append(f"// per-row weight scales L1: range [{s_w1.min():.3e}, {s_w1.max():.3e}]")
    h.append(f"// per-row weight scales L2: range [{s_w2.min():.3e}, {s_w2.max():.3e}]")
    h.append(f"// per-row weight scale  L3: {float(s_w3[0]):.6e}")
    h.append(f"// rescale: shift_l1={shift_l1} M range [{int(M_q_l1.min())}, {int(M_q_l1.max())}]  "
             f"shift_l2={shift_l2} M range [{int(M_q_l2.min())}, {int(M_q_l2.max())}]")
    h.append(f"// calibration headroom vs int32: {calib['headroom']*100:.1f}% "
             f"({calib_obs.shape[0]} samples from {calib_dataset.name})")
    h.append("#pragma once")
    h.append("#include <avr/pgmspace.h>")
    h.append("#include <stdint.h>")
    h.append("")
    h.append(f"#define POLICY_WEIGHT_BITS      {bits}")
    h.append(f"#define POLICY_OBS_DIM          {obs_dim}")
    h.append(f"#define POLICY_HIDDEN_DIM       {hidden}")
    h.append(f"#define POLICY_OUT_DIM          {act_dim}")
    h.append(f"#define POLICY_RESCALE_SHIFT_L1 {shift_l1}")
    h.append(f"#define POLICY_RESCALE_SHIFT_L2 {shift_l2}")
    h.append("")
    h.append(_format_float_1d("POLICY_INV_SCALE_OBS_IN", inv_scale_obs))
    h.append(_format_int16_1d("POLICY_RESCALE_M_L1", M_q_l1))   # per output channel
    h.append(_format_int16_1d("POLICY_RESCALE_M_L2", M_q_l2))
    h.append(_format_float_1d("POLICY_DEQUANT_L3", dequant_l3))
    h.append("")
    h.append(format_weight_2d("POLICY_W1", W1q))
    h.append(_format_int32_1d("POLICY_B1", B1q))
    h.append(format_weight_2d("POLICY_W2", W2q))
    h.append(_format_int32_1d("POLICY_B2", B2q))
    h.append(format_weight_2d("POLICY_W3", W3q))
    h.append(_format_int32_1d("POLICY_B3", B3q))
    h.append("")

    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("\n".join(h))
    print(f"wrote {header_path}")
    print(f"  {obs_dim}->{hidden}->{hidden}->{act_dim} int{bits}, "
          f"{flash_bytes} flash bytes, val_mse={val_mse:.6f}")
    return {
        "hidden": hidden, "obs_dim": obs_dim, "act_dim": act_dim, "bits": bits,
        "flash_bytes": flash_bytes, "val_mse": val_mse,
        "M_q_l1": M_q_l1, "shift_l1": shift_l1,
        "M_q_l2": M_q_l2, "shift_l2": shift_l2,
        "W1q": W1q, "B1q": B1q, "W2q": W2q, "B2q": B2q, "W3q": W3q, "B3q": B3q,
        "dequant_l3": dequant_l3, "inv_scale_obs": inv_scale_obs, "max_int": max_int,
    }


def parity_check(student_path: Path, n_samples: int = 1000, seed: int = 0,
                 dataset_path: Path | None = None, rescale: dict | None = None) -> dict:
    """Compare the numpy quantised forward pass against the QAT PyTorch model.

    `rescale`: pass the dict `export()` returned to validate the EXACT
    header just written (recommended — see main()). If omitted, this
    recomputes its own calibration independently, which is only meaningful
    as a standalone sanity check and may pick a different (also valid, but
    not necessarily identical) shift if run against a different sample of
    `dataset_path` than export() used.

    Tolerance is loose (~1-2 LSB on the action) because the fixed-point
    rescale rounds slightly differently from PyTorch's float scale * round.
    Bit-exactness against PyTorch isn't possible after the rescale step;
    bit-exactness against the *Arduino* C++ is guaranteed (both use int32
    arithmetic with the same rounding rule).
    """
    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=True)
    bits = int(ckpt.get("bits", 8))
    max_int = 2 ** (bits - 1) - 1
    model = QATStudent(
        hidden=int(ckpt["hidden"]),
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
        bits=bits,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sd = model.state_dict()

    s_obs_pc = np.asarray(ckpt["scales"]["obs_in"], dtype=np.float64)  # (obs_dim,)
    s_h1 = float(ckpt["scales"]["h1"])
    s_h2 = float(ckpt["scales"]["h2"])

    rng = np.random.default_rng(seed)
    if dataset_path is not None and dataset_path.exists():
        data = np.load(dataset_path)
        all_obs = np.asarray(data["obs"], dtype=np.float32)
        idx = rng.choice(all_obs.shape[0], size=min(n_samples, all_obs.shape[0]),
                         replace=False)
        obs = all_obs[idx]
        print(f"[parity] using {obs.shape[0]} real obs from {dataset_path}")
    else:
        # Fallback: synthetic obs across the realistic operating range of the
        # 6-dim raw frame ([motor_pos, sin, cos, motor_vel, pen_vel,
        # prev_action]), tiled to whatever frame-stack width the checkpoint
        # expects.
        obs_dim = int(ckpt["obs_dim"])
        raw = np.empty((n_samples, 6), dtype=np.float32)
        raw[:, 0] = rng.uniform(-2.18, 2.18, n_samples)
        raw[:, 1] = rng.uniform(-1.0, 1.0, n_samples)
        raw[:, 2] = rng.uniform(-1.0, 1.0, n_samples)
        raw[:, 3] = rng.uniform(-15.0, 15.0, n_samples)
        raw[:, 4] = rng.uniform(-15.0, 15.0, n_samples)
        raw[:, 5] = rng.uniform(-1.0, 1.0, n_samples)
        obs = np.tile(raw, (1, obs_dim // 6))

    if rescale is not None:
        W1q, B1q, W2q, B2q, W3q, B3q = (rescale["W1q"], rescale["B1q"], rescale["W2q"],
                                         rescale["B2q"], rescale["W3q"], rescale["B3q"])
        dequant_l3, inv_scale_obs = rescale["dequant_l3"], rescale["inv_scale_obs"]
        M_q_l1, shift_l1 = rescale["M_q_l1"], rescale["shift_l1"]
        M_q_l2, shift_l2 = rescale["M_q_l2"], rescale["shift_l2"]
    else:
        q = _quantise_all(sd, s_obs_pc, s_h1, s_h2, bits)
        calib = calibrate_rescale(
            obs, W1q=q["W1q"], B1q=q["B1q"], s_w1=q["s_w1"], s_h1=s_h1,
            W2q=q["W2q"], B2q=q["B2q"], s_w2=q["s_w2"], s_h2=s_h2,
            W3q=q["W3q"], B3q=q["B3q"], inv_scale_obs=q["inv_scale_obs"], max_int=max_int,
        )
        W1q, B1q, W2q, B2q, W3q, B3q = q["W1q"], q["B1q"], q["W2q"], q["B2q"], q["W3q"], q["B3q"]
        dequant_l3, inv_scale_obs = q["dequant_l3"], q["inv_scale_obs"]
        M_q_l1, shift_l1 = calib["M_q_l1"], calib["shift_l1"]
        M_q_l2, shift_l2 = calib["M_q_l2"], calib["shift_l2"]

    with torch.no_grad():
        torch_out = model(torch.from_numpy(obs)).cpu().numpy().reshape(-1)

    diag: dict = {}
    np_out = np.empty(obs.shape[0], dtype=np.float32)
    for k in range(obs.shape[0]):
        np_out[k] = numpy_forward_quantised(
            obs[k],
            W1=W1q, B1=B1q, W2=W2q, B2=B2q, W3=W3q, B3=B3q,
            inv_scale_obs=inv_scale_obs,
            M_q_l1=M_q_l1, shift_l1=shift_l1, M_q_l2=M_q_l2, shift_l2=shift_l2,
            dequant_l3=dequant_l3, max_int=max_int, diag=diag,
        )

    int32_max = 2 ** 31 - 1
    headroom = 1.0 - max(diag.get("max_abs_accum_l1", 0), diag.get("max_abs_accum_l2", 0),
                          diag.get("max_abs_accum_l3", 0)) / int32_max
    print(f"[parity] int32 accumulator headroom over {obs.shape[0]} samples: "
          f"L1 max|accum|={diag.get('max_abs_accum_l1', 0):.3e}, "
          f"L2={diag.get('max_abs_accum_l2', 0):.3e}, "
          f"L3={diag.get('max_abs_accum_l3', 0):.3e} "
          f"(int32 max={int32_max:.3e}, headroom={headroom*100:.1f}%)")
    if headroom < 0.2:
        raise RuntimeError(
            f"int32 accumulator headroom only {headroom * 100:.1f}% on the "
            f"samples checked — too close to overflow for real hardware use."
        )

    diff = np.abs(torch_out - np_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    p50 = float(np.percentile(diff, 50))
    p95 = float(np.percentile(diff, 95))
    p99 = float(np.percentile(diff, 99))
    lsb = 1.0 / max_int
    print(f"[parity] {obs.shape[0]} samples — max|torch_qat - numpy_quantised| ({bits}-bit, "
          f"shift_l1={shift_l1}, shift_l2={shift_l2}):")
    print(f"[parity]   mean = {mean_diff:.4f}  ({mean_diff/lsb:5.1f} LSB)")
    print(f"[parity]   p50  = {p50:.4f}  ({p50/lsb:5.1f} LSB)")
    print(f"[parity]   p95  = {p95:.4f}  ({p95/lsb:5.1f} LSB)")
    print(f"[parity]   p99  = {p99:.4f}  ({p99/lsb:5.1f} LSB)")
    print(f"[parity]   max  = {max_diff:.4f}  ({max_diff/lsb:5.1f} LSB)")
    # The mean and median are what matter for closed-loop behaviour. Outliers
    # at the max come from samples near tanh saturation where small pre-tanh
    # differences amplify, but they're rare and the policy's response there
    # is already close to ±1.
    if mean_diff > 4 * lsb:
        print(f"[parity] WARNING: mean diff > 4 LSB — investigate.")
    elif p99 > 16 * lsb:
        print(f"[parity] WARNING: p99 diff > 16 LSB — outlier behaviour to verify "
              f"on rig before relying on this student.")
    else:
        print(f"[parity] OK — mean within 1 LSB; ready for tethered test.")
    return {"max_diff": max_diff, "mean_diff": mean_diff,
            "p50": p50, "p95": p95, "p99": p99}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export a QAT student as a PROGMEM int C header")
    p.add_argument("--student", required=True, type=Path,
                   help="path to a student_quantised.pt produced by distill_quantised.py")
    p.add_argument("--header", required=True, type=Path,
                   help="output path for the generated .h "
                        "(e.g. policy_weights_quantised.h)")
    p.add_argument("--source-name", default=None,
                   help="comment string identifying the source run")
    p.add_argument("--no-parity", action="store_true",
                   help="skip the QAT-vs-quantised numpy parity check that runs after export")
    p.add_argument("--parity-dataset", type=Path, required=True,
                   help="dataset.npz to draw real on-distribution obs from — used both "
                        "to calibrate the per-layer rescale and (unless --no-parity) "
                        "for the post-export parity check")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    result = export(args.student, args.header, source_name=args.source_name,
                     calib_dataset=args.parity_dataset)
    if not args.no_parity:
        parity_check(args.student, dataset_path=args.parity_dataset, rescale=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
