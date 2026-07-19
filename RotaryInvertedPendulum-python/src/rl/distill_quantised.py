"""QAT (quantisation-aware training) variant of distill.py.

Produces an int8-deployable student MLP for the Arduino Nano. Format choice
rationale + AVR hardware reasoning live in `docs/quantisation.md`.

Pipeline overview (mirrors distill.py but with FakeQuant inserted):

    1. Load the float student (warmstart). The QAT model architecture is
       identical; we re-use the trained weights and let QAT fine-tune them
       to be int8-rounding-robust.
    2. Build the QAT model: same Linear layers + ReLU + tanh, with
       FakeQuantise wrappers around weights and activations. Activation
       scales are tracked via EMA of the per-batch absolute-max during
       training.
    3. Fine-tune on the same (obs, action_target) dataset that the float
       student trained on. ~100-200 epochs is plenty since we're starting
       from already-trained float weights.
    4. Save state-dict + activation scales. `export_weights_quantised.py`
       reads this and emits the C header.

Quantisation scheme: symmetric per-tensor int8, no zero point. Same scheme
the embedded-NN ecosystem (TFLite Micro, ONNX-RT mobile) standardises on.

    quantise(x, scale)   = clamp(round(x / scale), -127, 127)
    dequantise(q, scale) = q * scale

Usage:
    python distill_quantised.py \\
        --float-student runs/<run>/distill_h16_real_only/student.pt \\
        --dataset       runs/<run>/distill_h16_real_only/dataset.npz \\
        --out-dir       runs/<run>/distill_h16_quantised

The dataset is the .npz produced by `distill.py` (obs + action_target
arrays). We re-use it instead of re-running the teacher; the QAT step
is purely "robustify the existing student to int8 rounding".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distill import StudentMLP


# ---------------------------------------------------------------------------
# FakeQuant — straight-through int8 simulation
# ---------------------------------------------------------------------------

class FakeQuantSTE(torch.autograd.Function):
    """Symmetric per-tensor quantisation with straight-through gradient.

    Forward: x → clamp(round(x / scale), -max_int, max_int) * scale.
    Backward: gradient is passed through unchanged (the round() has zero
    gradient everywhere it's defined, so the STE pretends it's identity).

    `max_int` defaults to 127 (int8). For biases we use a much larger
    value so they're rounded to the grid but not clamped — the deployed
    int8 path stores biases as int32, which is effectively unclamped at
    the magnitudes we see in practice.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor,
                max_int: float = 127.0) -> torch.Tensor:
        scale = scale.clamp(min=1e-12)
        q = torch.round(x / scale).clamp(-max_int, max_int)
        return q * scale

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None


def fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8 fake-quant: round + clamp to [-127, 127]."""
    return FakeQuantSTE.apply(x, scale, 127.0)


def fake_quant_int32(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int32 fake-quant: round to the grid, ~no clamping. Used for biases —
    they live in the int32 accumulator on deploy, so int8 saturation is wrong.
    Limit set to 2**30 just as a safety net; biases are O(10²) at most.
    """
    return FakeQuantSTE.apply(x, scale, float(2**30))


# ---------------------------------------------------------------------------
# Activation observer — EMA of absolute-max during training
# ---------------------------------------------------------------------------

class MaxAbsObserver(nn.Module):
    """Tracks the running per-tensor |max| of activations via EMA.

    During training: updates the EMA every forward pass. The stored value
    is used to derive a quantisation scale (= |max| / 127).
    During eval: uses the frozen EMA value, so scales don't drift mid-eval.
    """

    def __init__(self, ema_decay: float = 0.99):
        super().__init__()
        self.ema_decay = ema_decay
        # Initialised to a tiny non-zero so the first forward pass doesn't
        # produce divide-by-zero in fake_quant.
        self.register_buffer("max_abs", torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                cur = x.detach().abs().max()
                # First update: just take the current value, otherwise the
                # 1e-3 init dominates for many epochs.
                if self.max_abs.item() <= 1e-3:
                    self.max_abs.copy_(cur)
                else:
                    self.max_abs.mul_(self.ema_decay).add_(cur, alpha=1.0 - self.ema_decay)
        return x

    @property
    def scale(self) -> torch.Tensor:
        return self.max_abs / 127.0


class PerChannelMaxAbsObserver(nn.Module):
    """Per-channel variant: tracks |max| separately along the last dimension.

    For an input of shape (batch, n_dim), tracks an EMA of max(|x|) over
    the batch axis, separately per dim. Used for the network's input layer
    where different obs dims have wildly different ranges (motor_pos ±2.18
    vs pen_vel ±30) — sharing one scale crushes the small-range dims to a
    handful of representable values right at the equilibrium.
    """

    def __init__(self, n_channels: int, ema_decay: float = 0.99):
        super().__init__()
        self.ema_decay = ema_decay
        self.register_buffer("max_abs", torch.full((n_channels,), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                # x shape: (batch, n_dim) — reduce over batch only.
                cur = x.detach().abs().amax(dim=0)
                # Per-channel first-update: replace tiny-init values with the
                # observed value (otherwise the EMA crawls up from 1e-3).
                init_mask = self.max_abs <= 1e-3
                self.max_abs[init_mask] = cur[init_mask]
                # Standard EMA on the rest.
                update_mask = ~init_mask
                if update_mask.any():
                    self.max_abs[update_mask] = (
                        self.ema_decay * self.max_abs[update_mask]
                        + (1.0 - self.ema_decay) * cur[update_mask]
                    )
        return x

    @property
    def scale(self) -> torch.Tensor:
        return self.max_abs / 127.0  # shape: (n_channels,)


# ---------------------------------------------------------------------------
# QAT student — same architecture as StudentMLP, with FakeQuant in the loop
# ---------------------------------------------------------------------------

class QATStudent(nn.Module):
    """5 → H → H → 1 MLP with int8 FakeQuant on weights + activations.

    Activation observers are placed at:
      - input (the obs vector)
      - after layer 1 ReLU
      - after layer 2 ReLU

    The pre-tanh activation is NOT quantised: the deployed int8 path
    dequantises the int32 layer-3 accumulator directly to float and
    applies float tanh (one libm call per inference, ~200 cycles —
    negligible). Quantising pre-tanh in QAT would force the trained
    weights to compensate for a precision loss the deploy never sees,
    which made the QAT-vs-int8 parity worse rather than better.
    """

    def __init__(self, hidden: int = 16, obs_dim: int = 5, act_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)
        # Per-channel observer for the input — the obs dims have very
        # different ranges (motor_pos ±2.18 vs pen_vel ±30), and a shared
        # scale crushes the small-range dims at the equilibrium. The
        # canonical "make int8 work for control" fix.
        self.obs_in = PerChannelMaxAbsObserver(obs_dim)
        # Hidden activations are uniform across channels by virtue of being
        # post-Linear+ReLU, so per-tensor scaling is fine for them.
        self.obs_h1 = MaxAbsObserver()
        self.obs_h2 = MaxAbsObserver()
        self.hidden = hidden
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @staticmethod
    def _q_weight_per_row(w: torch.Tensor) -> torch.Tensor:
        # Per-output-channel scaling: each row of W gets its own scale.
        # Standard TFLite-Micro pattern. For a Linear layer y = Wx + b,
        # the rescale to next layer's int8 then becomes per-output-channel
        # too — the M_q15 array on Arduino has H entries instead of one.
        # Avoids wasting precision on rows with smaller weight magnitudes.
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0  # (out, 1)
        return fake_quant(w, scale)

    @staticmethod
    def _q_bias_per_row(b: torch.Tensor, w_scale_per_row: torch.Tensor,
                       x_scale: torch.Tensor) -> torch.Tensor:
        # Bias scale is per output channel: s_b[i] = s_w[i] * s_x.
        # For per-channel input, x_scale is a tensor; we use its mean
        # magnitude here as the effective scalar, but because the absorbing
        # happens at export, this only affects QAT-time rounding noise.
        # In practice using the mean is close enough and stable.
        if x_scale.dim() > 0:
            x_scale = x_scale.mean()  # collapse per-channel input scale to scalar for bias
        bias_scale = (w_scale_per_row.squeeze(-1) * x_scale).clamp(min=1e-12)  # (out,)
        return fake_quant_int32(b, bias_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-channel input quantisation: each obs dim gets its own scale.
        self.obs_in(x)
        s_obs = self.obs_in.scale          # (obs_dim,)
        x_q = fake_quant(x, s_obs)         # snap each dim to its own int8 grid

        # ============ Layer 1 — input scale absorbed into weights ============
        # Export computes W_eff_int[i,j] = round(W[i,j] * s_obs[j] / s_w_eff[i])
        # and quantises per row. To make QAT match deploy bit-for-bit, we
        # do the same absorbing here: form W_eff = W * s_obs (broadcast over
        # rows), then per-row fake-quant. Then the matmul uses x_int_signal
        # = x_q / s_obs (integer-valued floats in [-127, 127]) so the
        # accumulator semantics match deploy: accum * s_w_eff[i] + bias.
        W_eff = self.fc1.weight * s_obs                                       # (H, O)
        s_w1_eff = W_eff.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0  # (H, 1)
        W_eff_q = fake_quant(W_eff, s_w1_eff)                                 # rounds per row
        s_b1 = s_w1_eff.squeeze(-1).clamp(min=1e-12)                          # (H,)
        b1 = fake_quant_int32(self.fc1.bias, s_b1)
        x_int_signal = x_q / s_obs                                            # int-valued in float
        x = F.linear(x_int_signal, W_eff_q, b1)
        x = F.relu(x)
        self.obs_h1(x)
        s_h1 = self.obs_h1.scale
        x = fake_quant(x, s_h1)

        # ============ Layer 2 — per-row weights, per-tensor input ============
        s_w2 = self.fc2.weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        w2_q = fake_quant(self.fc2.weight, s_w2)
        s_b2 = (s_w2.squeeze(-1) * s_h1).clamp(min=1e-12)
        b2 = fake_quant_int32(self.fc2.bias, s_b2)
        x = F.linear(x, w2_q, b2)
        x = F.relu(x)
        self.obs_h2(x)
        s_h2 = self.obs_h2.scale
        x = fake_quant(x, s_h2)

        # ============ Layer 3 — per-row weights, per-tensor input, float→tanh =====
        s_w3 = self.fc3.weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        w3_q = fake_quant(self.fc3.weight, s_w3)
        s_b3 = (s_w3.squeeze(-1) * s_h2).clamp(min=1e-12)
        b3 = fake_quant_int32(self.fc3.bias, s_b3)
        x = F.linear(x, w3_q, b3)
        return torch.tanh(x)


# ---------------------------------------------------------------------------
# Warmstart helpers — copy float student weights into the QAT model
# ---------------------------------------------------------------------------

def warmstart_from_float(qat: QATStudent, float_student: StudentMLP) -> None:
    """Copy fc1/fc2/fc3 weights+biases from the float student into the QAT model."""
    qat.fc1.load_state_dict(float_student.fc1.state_dict())
    qat.fc2.load_state_dict(float_student.fc2.state_dict())
    qat.fc3.load_state_dict(float_student.fc3.state_dict())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_qat(
    qat: QATStudent,
    obs: np.ndarray,
    target: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    idx = rng.permutation(n)
    n_val = int(val_frac * n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
    tgt_t = torch.from_numpy(target.astype(np.float32)).to(device)
    train_obs, train_tgt = obs_t[train_idx], tgt_t[train_idx]
    val_obs, val_tgt = obs_t[val_idx], tgt_t[val_idx]

    qat = qat.to(device)
    opt = torch.optim.Adam(qat.parameters(), lr=lr)
    n_train = train_obs.shape[0]

    last_val = float("nan")
    for epoch in range(1, epochs + 1):
        qat.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        seen = 0
        for i in range(0, n_train, batch_size):
            sel = perm[i : i + batch_size]
            x = train_obs[sel]
            y = train_tgt[sel]
            pred = qat(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * x.shape[0]
            seen += x.shape[0]
        train_loss = train_loss_sum / max(1, seen)

        qat.eval()
        with torch.no_grad():
            val_pred = qat(val_obs)
            val_loss = F.mse_loss(val_pred, val_tgt).item()
        last_val = val_loss
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"[qat] epoch {epoch:3d}/{epochs}  "
                  f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")
    return {"val_mse": last_val}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_qat_checkpoint(qat: QATStudent, out_path: Path, val_mse: float) -> None:
    """Save model weights + the activation-scale observers' EMA state."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": qat.state_dict(),
        "hidden": qat.hidden,
        "obs_dim": qat.obs_dim,
        "act_dim": qat.act_dim,
        "val_mse": val_mse,
        # Convenience: surface activation scales as plain floats so
        # export_weights_quantised.py can read them without touching nn.Module.
        "scales": {
            "obs_in": qat.obs_in.scale.detach().cpu().numpy().tolist(),  # per-channel list
            "h1": float(qat.obs_h1.scale.item()),
            "h2": float(qat.obs_h2.scale.item()),
        },
        "max_abs": {
            "obs_in": qat.obs_in.max_abs.detach().cpu().numpy().tolist(),
            "h1": float(qat.obs_h1.max_abs.item()),
            "h2": float(qat.obs_h2.max_abs.item()),
        },
    }
    torch.save(payload, out_path)
    print(f"[qat] saved -> {out_path}, val_mse={val_mse:.6f}")
    obs_scales = payload['scales']['obs_in']
    obs_scales_str = ', '.join(f'{s:.4f}' for s in obs_scales)
    print(f"[qat] activation scales:")
    print(f"[qat]   obs_in (per-channel): [{obs_scales_str}]")
    print(f"[qat]   h1={payload['scales']['h1']:.6f}  h2={payload['scales']['h2']:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QAT distillation of a float student to int8")
    p.add_argument("--float-student", required=True, type=Path,
                   help="path to the float student.pt produced by distill.py")
    p.add_argument("--dataset", required=True, type=Path,
                   help="path to the dataset.npz produced by distill.py "
                        "(re-used as-is; QAT just robustifies the existing student)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="output directory for the quantised .pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="default 3e-4: lower than distill.py's 1e-3 because "
                        "we're fine-tuning a converged model, not training "
                        "from scratch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    print(f"[qat] loading float student: {args.float_student}")
    ckpt = torch.load(str(args.float_student), map_location=args.device, weights_only=True)
    hidden = int(ckpt["hidden"])
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    print(f"[qat] architecture: {obs_dim} -> {hidden} -> {hidden} -> {act_dim}, "
          f"float val_mse={ckpt.get('val_mse', float('nan')):.6f}")

    float_student = StudentMLP(hidden=hidden, obs_dim=obs_dim, act_dim=act_dim)
    float_student.load_state_dict(ckpt["state_dict"])

    qat = QATStudent(hidden=hidden, obs_dim=obs_dim, act_dim=act_dim)
    warmstart_from_float(qat, float_student)
    print(f"[qat] warmstarted QAT model from float student")

    print(f"[qat] loading dataset: {args.dataset}")
    data = np.load(args.dataset)
    obs = np.asarray(data["obs"], dtype=np.float32)
    target = np.asarray(data["action_target"], dtype=np.float32)
    print(f"[qat] {obs.shape[0]} samples")

    t0 = time.time()
    metrics = train_qat(
        qat, obs, target,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, device=args.device,
    )
    dt = time.time() - t0
    print(f"[qat] training took {dt:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_qat_checkpoint(qat, args.out_dir / "student_quantised.pt", metrics["val_mse"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
