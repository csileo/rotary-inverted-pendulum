"""FFT analysis of a deploy log to test the resonance-exploit hypothesis.

If the policy is doing Kapitza-style parametric stabilisation, the motor
velocity during balance will be dominated by a SHARP spectral peak at
the resonance frequency (and possibly harmonics). If the policy is doing
broadband corrective feedback control, the motor velocity spectrum will
be broad/flat with no clear dominant frequency.

Usage:
    python fft_deploy.py /tmp/2026-05-20_16-13_75hz.npz
    python fft_deploy.py /tmp/a.npz /tmp/b.npz   # overlay
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PENDULUM_NATURAL_F_HZ = 1.82  # from sysid_params (T_small_amp ≈ 0.55 s)


def fft_signal(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of x(t)."""
    # Use uniform dt: median is robust to startup jitter.
    dt = float(np.median(np.diff(t)))
    n = len(x)
    # Remove DC bias.
    x = x - x.mean()
    # Hann window to reduce spectral leakage.
    window = np.hanning(n)
    x_windowed = x * window
    # FFT; scale to amplitude.
    Y = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, d=dt)
    # Compensate Hann window's amplitude reduction (~50%).
    amp = np.abs(Y) * (2.0 / window.sum())
    return freqs, amp


def _theta_to_upright(pen_pos: np.ndarray) -> np.ndarray:
    e = (pen_pos - np.pi + np.pi) % (2 * np.pi) - np.pi
    return e


def find_balance_window(pen_pos: np.ndarray, t: np.ndarray,
                        threshold_rad: float = 0.3,
                        settle_s: float = 1.0) -> tuple[int, int]:
    """First steady-state balance window: from `settle_s` after first
    near-upright crossing, until either a fall or end of log.
    """
    err = _theta_to_upright(pen_pos)
    near = np.abs(err) < threshold_rad
    if not near.any():
        return 0, 0
    i0 = int(np.argmax(near))
    t_start = t[i0] + settle_s
    i_start = int(np.searchsorted(t, t_start))
    return i_start, len(t)


def analyze(path: Path) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    t = (d["time_us"] - d["time_us"][0]) / 1e6
    motor_vel = d["motor_vel_rad_s"].astype(np.float64)
    motor_pos = d["motor_pos_rad"].astype(np.float64)
    pen_pos = d["pendulum_pos_rad"].astype(np.float64)
    accel_cmd = d["accel_cmd_rad_s2"].astype(np.float64)

    i0, i1 = find_balance_window(pen_pos, t)
    n = i1 - i0
    if n < 100:
        return dict(path=str(path), error=f"insufficient balance window ({n} samples)")

    sel = slice(i0, i1)
    sample_hz = (n - 1) / (t[sel][-1] - t[sel][0])

    f_mv, A_mv = fft_signal(t[sel], motor_vel[sel])
    f_mp, A_mp = fft_signal(t[sel], motor_pos[sel])
    f_ac, A_ac = fft_signal(t[sel], accel_cmd[sel])

    # Skip very low freqs (the windowed DC residual) when picking peaks.
    valid = f_mv > 0.5
    peak_idx = int(np.argmax(A_mv[valid]) + np.argmax(valid))
    peak_f = float(f_mv[peak_idx])
    peak_amp = float(A_mv[peak_idx])

    # Compute peak sharpness: ratio of peak amplitude to the median spectrum
    # outside the peak band. High ratio = sharp peak = resonance. Low = broadband.
    not_peak = (f_mv < peak_f * 0.5) | (f_mv > peak_f * 1.5)
    not_peak &= valid
    median_off_peak = float(np.median(A_mv[not_peak])) if not_peak.any() else 1e-9
    peak_ratio = peak_amp / max(median_off_peak, 1e-9)

    return dict(
        path=str(path),
        balance_samples=int(n),
        sample_hz=float(sample_hz),
        peak_f=peak_f,
        peak_amp=peak_amp,
        peak_ratio=peak_ratio,
        median_off_peak=median_off_peak,
        f_mv=f_mv, A_mv=A_mv,
        f_mp=f_mp, A_mp=A_mp,
        f_ac=f_ac, A_ac=A_ac,
    )


def print_report(result: dict) -> None:
    print(f"\n  {Path(result['path']).name}")
    if "error" in result:
        print(f"    {result['error']}")
        return
    print(f"    balance samples:     {result['balance_samples']} "
          f"({result['sample_hz']:.0f} Hz)")
    print(f"    motor_vel peak freq: {result['peak_f']:.2f} Hz")
    print(f"    motor_vel peak amp:  {result['peak_amp']:.3f} rad/s")
    print(f"    peak/off-peak ratio: {result['peak_ratio']:.1f}× "
          f"(>10× → sharp resonance; <3× → broadband)")
    f_ratio = result['peak_f'] / PENDULUM_NATURAL_F_HZ
    print(f"    f_peak / f_pendulum: {f_ratio:.2f}×  "
          f"(natural f_pendulum ≈ {PENDULUM_NATURAL_F_HZ:.2f} Hz)")
    if result['peak_ratio'] > 10:
        verdict = "SHARP peak → likely Kapitza-style resonance exploit"
    elif result['peak_ratio'] > 3:
        verdict = "moderate peak → mixed strategy"
    else:
        verdict = "broadband → corrective feedback, no resonance"
    print(f"    verdict:             {verdict}")


def plot_spectra(results: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    valid = [r for r in results if "error" not in r]
    if not valid:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = ["C0", "C3", "C2", "C4"]
    for i, r in enumerate(valid):
        c = colors[i % len(colors)]
        label = Path(r["path"]).name
        axes[0].semilogy(r["f_mv"], r["A_mv"], c, lw=1, label=label)
        axes[1].semilogy(r["f_mp"], r["A_mp"], c, lw=1, label=label)
        axes[2].semilogy(r["f_ac"], r["A_ac"], c, lw=1, label=label)

    for ax in axes:
        ax.axvline(PENDULUM_NATURAL_F_HZ, color="k", ls=":", alpha=0.6,
                   label=f"f_pendulum = {PENDULUM_NATURAL_F_HZ:.2f} Hz" if ax is axes[0] else None)
        ax.axvline(2 * PENDULUM_NATURAL_F_HZ, color="gray", ls=":", alpha=0.4,
                   label="2× f_pendulum" if ax is axes[0] else None)
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("motor_vel amplitude\n(rad/s)")
    axes[1].set_ylabel("motor_pos amplitude\n(rad)")
    axes[2].set_ylabel("accel_cmd amplitude\n(rad/s²)")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_xlim(0, 30)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Deploy spectra — balance-window FFT", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    results = [analyze(Path(pp)) for pp in args.paths]
    print(f"Pendulum natural frequency (small-amplitude): {PENDULUM_NATURAL_F_HZ:.2f} Hz")
    for r in results:
        print_report(r)

    if args.out:
        out = Path(args.out)
    else:
        out = Path(args.paths[0]).with_name(Path(args.paths[0]).stem + "_fft.png")
    plot_spectra(results, out)
    print(f"\nSpectrum plot: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
