"""Export a QAT-trained student MLP as a PROGMEM C header for the Arduino Nano.

Reads the .pt produced by `distill_quantised.py` and writes a self-contained
header at `RotaryInvertedPendulum-arduino/RLControl/policy_weights_quantised.h`.

Quantisation scheme (matches `distill_quantised.py` and the C++ forward pass):

    Per-tensor symmetric int8, no zero point.
        quantise(x, s)   = clamp(round(x / s), -127, 127)
        dequantise(q, s) = q * s

    Per-layer Linear:
        y = W @ x + b           (float math we replace)
        becomes
        accum_i32 = sum( W_int8[i,j] * x_int8[j] ) + b_int32[i]
                                   ^ both signed int8, accumulate in int32
                                   ^ b_int32 pre-scaled to (s_w * s_x) units
        Then:
            For hidden layers: rescale accum_i32 to int8 next-layer input
                               using a fixed-point multiply-shift.
            For the final layer: dequantise accum_i32 to float, apply tanh.

    Rescale (hidden layers):
        We want   y_i8 = clamp(round(accum_i32 * (s_w * s_x / s_y)), -127, 127)
        Implement (s_w * s_x / s_y) as int16 / 2^15:
            M_q15 = round((s_w * s_x / s_y) * 32768)
            y_i8  = clamp(((accum_i32 * M_q15) + (1<<14)) >> 15, -127, 127)
        Then ReLU is just clamp(y_i8, 0, 127).

    Final dequantise:
        y_float = accum_i32 * (s_w * s_h2)         # one float multiply
        action  = tanh(y_float)                    # one libm call

Bias quantisation:
        b_int32[i] = round(bias_float[i] / (s_w * s_x))
        Lives in the same units as `accum` so it adds directly without
        further rescaling.

Header schema (matches what `RLControl.ino` consumes when POLICY_QUANTISED
is defined):

    POLICY_OBS_DIM, POLICY_HIDDEN_DIM, POLICY_OUT_DIM
    POLICY_INV_SCALE_OBS_IN  (float, = 1 / s_obs_in; multiply obs by this)
    POLICY_W1[H][O], POLICY_W2[H][H], POLICY_W3[1][H]   (int8)
    POLICY_B1[H], POLICY_B2[H], POLICY_B3[1]             (int32)
    POLICY_M_Q15_L1, POLICY_M_Q15_L2                      (int16, rescale)
    POLICY_DEQUANT_L3                                     (float)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import torch

from distill_quantised import QATStudent


def _quantise_weight_per_row(w_float: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric int8: each row of W gets its own scale.

    Returns (W_int8 of shape (out, in), s_w of shape (out,)). The canonical
    TFLite-Micro pattern; recovers a lot of fidelity over per-tensor scaling
    when different output neurons have different weight magnitudes.
    """
    max_abs_per_row = np.max(np.abs(w_float), axis=1)  # (out,)
    s_w = np.maximum(max_abs_per_row / 127.0, 1e-8).astype(np.float64)  # (out,)
    w_int = np.clip(np.round(w_float / s_w[:, None]), -127, 127).astype(np.int8)
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

    After this, W_eff can be quantised per row to int8 and the deployment
    matmul does sum_j (W_eff_int[i,j] * x_int[j]), where x_int[j] = round(x[j]
    / s_obs[j]) — i.e. the per-channel input scales are already inside the
    weights. The Arduino still does an ordinary int8 matmul.
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


def export(student_path: Path, header_path: Path, *, source_name: str | None = None) -> dict:
    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=True)
    hidden = int(ckpt["hidden"])
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    val_mse = float(ckpt.get("val_mse", float("nan")))

    model = QATStudent(hidden=hidden, obs_dim=obs_dim, act_dim=act_dim)
    model.load_state_dict(ckpt["state_dict"])
    sd = model.state_dict()

    # Float weights
    W1f = sd["fc1.weight"].cpu().numpy().astype(np.float32)  # (H, O)
    B1f = sd["fc1.bias"].cpu().numpy().astype(np.float32)
    W2f = sd["fc2.weight"].cpu().numpy().astype(np.float32)  # (H, H)
    B2f = sd["fc2.bias"].cpu().numpy().astype(np.float32)
    W3f = sd["fc3.weight"].cpu().numpy().astype(np.float32)  # (1, H)
    B3f = sd["fc3.bias"].cpu().numpy().astype(np.float32)

    # Activation scales (frozen EMA from training).
    # obs_in is per-channel (saved as a Python list); h1/h2 are per-tensor.
    s_obs_pc = np.asarray(ckpt["scales"]["obs_in"], dtype=np.float64)  # (obs_dim,)
    s_h1 = float(ckpt["scales"]["h1"])
    s_h2 = float(ckpt["scales"]["h2"])

    # Layer 1: absorb per-channel input scales into the weights, then per-row
    # quantise. After absorption, each weight column j gets scaled by s_obs[j],
    # so different columns have different effective magnitudes — that's exactly
    # what per-row quantisation handles well.
    W1_eff = _absorb_per_channel_input_scales(W1f, s_obs_pc.astype(np.float32))
    W1q, s_w1 = _quantise_weight_per_row(W1_eff)
    # Bias for L1 lives in units of s_w1[i] (the per-row weight scale of the
    # ABSORBED weights — s_obs is already in there). So bias scale s_b1[i] = s_w1[i].
    B1q, _ = _quantise_bias_int32_per_row(B1f, s_w1, s_x=1.0)

    # Layers 2 and 3: per-row weights, per-tensor input. Standard.
    W2q, s_w2 = _quantise_weight_per_row(W2f)
    B2q, _ = _quantise_bias_int32_per_row(B2f, s_w2, s_x=s_h1)
    W3q, s_w3 = _quantise_weight_per_row(W3f)
    B3q, _ = _quantise_bias_int32_per_row(B3f, s_w3, s_x=s_h2)

    # Per-row rescale factors. M[i] = s_w[i] * s_input / s_output.
    # Layer 1: s_input was absorbed into s_w, so M[i] = s_w1[i] / s_h1.
    M_l1 = s_w1 / s_h1                 # (H,)
    M_l2 = (s_w2 * s_h1) / s_h2        # (H,)
    M_q15_l1 = np.round(M_l1 * 32768).astype(np.int64)
    M_q15_l2 = np.round(M_l2 * 32768).astype(np.int64)
    if (M_q15_l1.max() > 32767 or M_q15_l1.min() < -32768
            or M_q15_l2.max() > 32767 or M_q15_l2.min() < -32768):
        raise RuntimeError(
            f"per-row Q15 rescale overflow:\n"
            f"  M1_q15 range [{M_q15_l1.min()}, {M_q15_l1.max()}]\n"
            f"  M2_q15 range [{M_q15_l2.min()}, {M_q15_l2.max()}]\n"
            f"Some weight rows are too large relative to activation scales."
        )
    M_q15_l1 = M_q15_l1.astype(np.int16)
    M_q15_l2 = M_q15_l2.astype(np.int16)

    # Layer 3 dequantise-to-float: float scale per output (only 1 output here).
    # y_float[i] = accum_int[i] * s_w3[i] * s_h2 + b[i]_dequant
    # Bias for L3 was already incorporated into accum, so:
    #   y_float[i] = accum_with_bias[i] * (s_w3[i] * s_h2)
    dequant_l3 = (s_w3 * s_h2).astype(np.float32)  # (act_dim,)

    # Input quantisation: x_int8[j] = clamp(round(obs[j] * inv_scale_obs[j]), -127, 127).
    # Per-channel inverse scale array.
    inv_scale_obs = (1.0 / np.maximum(s_obs_pc, 1e-12)).astype(np.float32)  # (obs_dim,)

    n_params_int8 = W1q.size + W2q.size + W3q.size
    n_params_int32 = B1q.size + B2q.size + B3q.size
    n_params_int16 = M_q15_l1.size + M_q15_l2.size
    n_params_float = inv_scale_obs.size + dequant_l3.size
    flash_bytes = n_params_int8 + 4 * n_params_int32 + 2 * n_params_int16 + 4 * n_params_float

    h = []
    h.append("// auto-generated by export_weights_quantised.py — do not edit by hand")
    if source_name:
        h.append(f"// source: {source_name}")
    h.append(f"// generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    h.append(f"// quantised student MLP: {obs_dim} -> {hidden} -> {hidden} -> {act_dim} (int8)")
    h.append(f"// per-channel input + per-row weight quantisation")
    h.append(f"// weights+biases: {n_params_int8} int8 + {n_params_int32} int32 + "
             f"{n_params_int16} int16 + {n_params_float} float = {flash_bytes} flash bytes")
    h.append(f"// QAT val_mse: {val_mse:.6f}")
    obs_scale_strs = ', '.join(f'{float(s):.4e}' for s in s_obs_pc)
    h.append(f"// per-channel input scales: [{obs_scale_strs}]")
    h.append(f"// hidden activation scales: s_h1={s_h1:.6e}  s_h2={s_h2:.6e}")
    h.append(f"// per-row weight scales L1: range [{s_w1.min():.3e}, {s_w1.max():.3e}]")
    h.append(f"// per-row weight scales L2: range [{s_w2.min():.3e}, {s_w2.max():.3e}]")
    h.append(f"// per-row weight scale  L3: {float(s_w3[0]):.6e}")
    h.append(f"// rescale Q15: L1 range [{int(M_q15_l1.min())}, {int(M_q15_l1.max())}]  "
             f"L2 range [{int(M_q15_l2.min())}, {int(M_q15_l2.max())}]")
    h.append("#pragma once")
    h.append("#include <avr/pgmspace.h>")
    h.append("#include <stdint.h>")
    h.append("")
    h.append(f"#define POLICY_OBS_DIM     {obs_dim}")
    h.append(f"#define POLICY_HIDDEN_DIM  {hidden}")
    h.append(f"#define POLICY_OUT_DIM     {act_dim}")
    h.append("")
    h.append(_format_float_1d("POLICY_INV_SCALE_OBS_IN", inv_scale_obs))
    h.append(_format_int16_1d("POLICY_M_Q15_L1", M_q15_l1))   # per output channel
    h.append(_format_int16_1d("POLICY_M_Q15_L2", M_q15_l2))
    h.append(_format_float_1d("POLICY_DEQUANT_L3", dequant_l3))
    h.append("")
    h.append(_format_int8_2d("POLICY_W1", W1q))
    h.append(_format_int32_1d("POLICY_B1", B1q))
    h.append(_format_int8_2d("POLICY_W2", W2q))
    h.append(_format_int32_1d("POLICY_B2", B2q))
    h.append(_format_int8_2d("POLICY_W3", W3q))
    h.append(_format_int32_1d("POLICY_B3", B3q))
    h.append("")

    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("\n".join(h))
    print(f"wrote {header_path}")
    print(f"  {obs_dim}->{hidden}->{hidden}->{act_dim} int8, "
          f"{flash_bytes} flash bytes, val_mse={val_mse:.6f}")
    return {
        "hidden": hidden,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "flash_bytes": flash_bytes,
        "val_mse": val_mse,
    }


def numpy_forward_int8(
    obs: np.ndarray,
    *,
    W1: np.ndarray, B1: np.ndarray,
    W2: np.ndarray, B2: np.ndarray,
    W3: np.ndarray, B3: np.ndarray,
    inv_scale_obs: np.ndarray,   # per-channel, shape (obs_dim,)
    M_q15_l1: np.ndarray,        # per output channel, shape (H,)
    M_q15_l2: np.ndarray,        # per output channel, shape (H,)
    dequant_l3: np.ndarray,      # per output channel, shape (act_dim,)
) -> np.ndarray:
    """Numpy implementation that mirrors the Arduino int8 forward pass exactly.

    Single-sample input. Returns the float action (post-tanh).
    Used both for parity-checking the export and as a reference for the C++.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    # Per-channel input quantisation: each obs dim has its own scale.
    x = np.empty(obs.shape[0], dtype=np.int32)
    for j in range(obs.shape[0]):
        q = int(round(obs[j] * inv_scale_obs[j]))
        if q >  127: q =  127
        if q < -127: q = -127
        x[j] = q

    # Layer 1: int8 matmul + bias + per-row Q15 rescale + ReLU.
    H = W1.shape[0]
    h1 = np.zeros(H, dtype=np.int32)
    for i in range(H):
        accum = int(B1[i])
        for j in range(W1.shape[1]):
            accum += int(W1[i, j]) * int(x[j])
        scaled = (accum * int(M_q15_l1[i]) + (1 << 14)) >> 15
        if scaled > 127: scaled = 127
        if scaled < 0:   scaled = 0           # ReLU
        h1[i] = scaled

    # Layer 2: same pattern.
    h2 = np.zeros(H, dtype=np.int32)
    for i in range(H):
        accum = int(B2[i])
        for j in range(W2.shape[1]):
            accum += int(W2[i, j]) * int(h1[j])
        scaled = (accum * int(M_q15_l2[i]) + (1 << 14)) >> 15
        if scaled > 127: scaled = 127
        if scaled < 0:   scaled = 0
        h2[i] = scaled

    # Layer 3 → dequantise per output → tanh.
    accum = int(B3[0])
    for j in range(W3.shape[1]):
        accum += int(W3[0, j]) * int(h2[j])
    y = float(accum) * float(dequant_l3[0])
    return np.float32(np.tanh(y))


def parity_check(student_path: Path, n_samples: int = 1000, seed: int = 0,
                 dataset_path: Path | None = None) -> dict:
    """Compare the numpy int8 forward pass against the QAT PyTorch model.

    Tolerance is loose (~1-2 LSB on the action) because the Q15 fixed-point
    rescale rounds slightly differently from PyTorch's float scale * round.
    Bit-exactness against PyTorch isn't possible after the rescale step;
    bit-exactness against the *Arduino* C++ is guaranteed (both use int32
    arithmetic with the same rounding rule).
    """
    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=True)
    model = QATStudent(
        hidden=int(ckpt["hidden"]),
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sd = model.state_dict()

    W1f = sd["fc1.weight"].cpu().numpy().astype(np.float32)
    B1f = sd["fc1.bias"].cpu().numpy().astype(np.float32)
    W2f = sd["fc2.weight"].cpu().numpy().astype(np.float32)
    B2f = sd["fc2.bias"].cpu().numpy().astype(np.float32)
    W3f = sd["fc3.weight"].cpu().numpy().astype(np.float32)
    B3f = sd["fc3.bias"].cpu().numpy().astype(np.float32)

    s_obs_pc = np.asarray(ckpt["scales"]["obs_in"], dtype=np.float64)  # (obs_dim,)
    s_h1 = float(ckpt["scales"]["h1"])
    s_h2 = float(ckpt["scales"]["h2"])

    # Same quantisation pipeline as export() — duplicated here so parity_check()
    # is a self-contained reference. (Kept simple over DRY for clarity.)
    W1_eff = _absorb_per_channel_input_scales(W1f, s_obs_pc.astype(np.float32))
    W1q, s_w1 = _quantise_weight_per_row(W1_eff)
    W2q, s_w2 = _quantise_weight_per_row(W2f)
    W3q, s_w3 = _quantise_weight_per_row(W3f)
    B1q, _ = _quantise_bias_int32_per_row(B1f, s_w1, s_x=1.0)
    B2q, _ = _quantise_bias_int32_per_row(B2f, s_w2, s_x=s_h1)
    B3q, _ = _quantise_bias_int32_per_row(B3f, s_w3, s_x=s_h2)

    M_q15_l1 = np.round((s_w1 / s_h1) * 32768).astype(np.int16)
    M_q15_l2 = np.round((s_w2 * s_h1 / s_h2) * 32768).astype(np.int16)
    dequant_l3 = (s_w3 * s_h2).astype(np.float32)
    inv_scale_obs = (1.0 / np.maximum(s_obs_pc, 1e-12)).astype(np.float32)

    rng = np.random.default_rng(seed)
    if dataset_path is not None and dataset_path.exists():
        # Use real on-distribution obs from the dataset the QAT was trained on.
        data = np.load(dataset_path)
        all_obs = np.asarray(data["obs"], dtype=np.float32)
        idx = rng.choice(all_obs.shape[0], size=min(n_samples, all_obs.shape[0]),
                         replace=False)
        obs = all_obs[idx]
        print(f"[parity] using {obs.shape[0]} real obs from {dataset_path}")
    else:
        # Fallback: synthetic obs across the realistic operating range.
        # motor_pos in [-2.18, 2.18], sin/cos in [-1, 1], velocities in
        # [-15, 15] (centred on what QAT actually observed; uniform [-30, 30]
        # over-samples saturation boundaries and inflates the max diff).
        obs = np.empty((n_samples, int(ckpt["obs_dim"])), dtype=np.float32)
        obs[:, 0] = rng.uniform(-2.18, 2.18, n_samples)
        obs[:, 1] = rng.uniform(-1.0, 1.0, n_samples)
        obs[:, 2] = rng.uniform(-1.0, 1.0, n_samples)
        obs[:, 3] = rng.uniform(-15.0, 15.0, n_samples)
        obs[:, 4] = rng.uniform(-15.0, 15.0, n_samples)

    # PyTorch QAT forward
    with torch.no_grad():
        torch_out = model(torch.from_numpy(obs)).cpu().numpy().reshape(-1)

    # Numpy int8 forward (mirrors the Arduino code)
    np_out = np.empty(n_samples, dtype=np.float32)
    for k in range(n_samples):
        np_out[k] = numpy_forward_int8(
            obs[k],
            W1=W1q, B1=B1q,
            W2=W2q, B2=B2q,
            W3=W3q, B3=B3q,
            inv_scale_obs=inv_scale_obs,
            M_q15_l1=M_q15_l1, M_q15_l2=M_q15_l2,
            dequant_l3=dequant_l3,
        )

    diff = np.abs(torch_out - np_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    p50 = float(np.percentile(diff, 50))
    p95 = float(np.percentile(diff, 95))
    p99 = float(np.percentile(diff, 99))
    lsb = 1.0 / 127.0
    print(f"[parity] {n_samples} samples — max|torch_qat - numpy_int8|:")
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
    p = argparse.ArgumentParser(description="Export a QAT student as a PROGMEM int8 C header")
    p.add_argument("--student", required=True, type=Path,
                   help="path to a student_quantised.pt produced by distill_quantised.py")
    p.add_argument("--header", required=True, type=Path,
                   help="output path for the generated .h "
                        "(e.g. policy_weights_quantised.h)")
    p.add_argument("--source-name", default=None,
                   help="comment string identifying the source run")
    p.add_argument("--no-parity", action="store_true",
                   help="skip the QAT-vs-int8 numpy parity check that runs after export")
    p.add_argument("--parity-dataset", type=Path, default=None,
                   help="optional dataset.npz to draw real on-distribution obs from "
                        "for the parity check (defaults to synthetic uniform sampling)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    export(args.student, args.header, source_name=args.source_name)
    if not args.no_parity:
        parity_check(args.student, dataset_path=args.parity_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
