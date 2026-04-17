#!/usr/bin/env python3
"""
Synthetic audio data generator -- Green Wave++

Generates labelled WAV files without any real microphone hardware:
  - Positive class: synthetic siren tones at various SNRs and bearings
  - Negative class: simulated road noise, white noise, and urban ambient

Output structure matches what AudioDataset expects:
    audio/data/processed/
        train/
            positive/  ← siren WAVs
            negative/  ← non-siren WAVs
        val/
            positive/
            negative/

This is intentionally signal-generator code, not production audio -- the
goal is to give the CRNN a reasonable starting distribution so it can be
fine-tuned on real siren recordings without needing a large synthetic set.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000   # 16 kHz, matches config.yaml


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def siren_wail(duration_s: float, base_hz: float = 750.0, mod_rate: float = 0.9) -> np.ndarray:
    """
    American-style wailing siren: slow FM sweep between base_hz and base_hz*1.5.
    mod_rate controls sweeps per second.
    """
    t     = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    sweep = np.sin(2 * np.pi * mod_rate * t)   # –1 ... +1

    # Frequency sweeps between base and base*1.5
    lo, hi   = base_hz, base_hz * 1.5
    inst_hz  = lo + (hi - lo) * (0.5 + 0.5 * sweep)
    phase    = 2 * np.pi * np.cumsum(inst_hz) / SR
    signal   = np.sin(phase)

    # Slight amplitude envelope to avoid abrupt start/stop
    env = np.ones_like(signal)
    fade = int(0.05 * SR)
    env[:fade]  = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (signal * env).astype(np.float32)


def yelp_siren(duration_s: float, rate_hz: float = 4.0) -> np.ndarray:
    """Fast on-off yelp pattern."""
    t      = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    toggle = (np.sin(2 * np.pi * rate_hz * t) > 0).astype(np.float32)
    carrier = np.sin(2 * np.pi * 1100.0 * t)
    return (carrier * toggle).astype(np.float32)


def phaser_siren(duration_s: float) -> np.ndarray:
    """European hi-lo phaser."""
    n    = int(SR * duration_s)
    t    = np.linspace(0, duration_s, n, endpoint=False)
    # Alternate between 400 Hz and 900 Hz at ~1 Hz
    inst = np.where(np.sin(2 * np.pi * 0.8 * t) > 0, 400.0, 900.0)
    phase = 2 * np.pi * np.cumsum(inst) / SR
    return np.sin(phase).astype(np.float32)


def road_noise(duration_s: float) -> np.ndarray:
    """Low-frequency coloured noise mimicking road rumble."""
    n    = int(SR * duration_s)
    wn   = np.random.randn(n)
    # Simple first-order low-pass: y[n] = α*y[n-1] + (1-α)*x[n]
    α    = 0.97
    out  = np.zeros(n, dtype=np.float32)
    prev = 0.0
    for i in range(n):
        prev   = α * prev + (1 - α) * wn[i]
        out[i] = prev
    return out / (np.abs(out).max() + 1e-8)


def crowd_noise(duration_s: float) -> np.ndarray:
    """Band-passed noise to simulate a crowd or traffic intersection."""
    n  = int(SR * duration_s)
    wn = np.random.randn(n).astype(np.float32)
    # Band-pass: rough IIR approximation for 200–2000 Hz
    out  = np.zeros(n, dtype=np.float32)
    prev = np.zeros(3, dtype=np.float32)
    for i in range(n):
        prev[0] = 0.9 * prev[0] + 0.1  * wn[i]
        prev[1] = 0.85 * prev[1] + 0.15 * (wn[i] - prev[0])
        out[i]  = prev[1]
    out /= np.abs(out).max() + 1e-8
    return out


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise to signal at the specified SNR (dB)."""
    # Align lengths
    n = min(len(signal), len(noise))
    s, no = signal[:n], noise[:n]

    sig_pwr   = np.mean(s  ** 2) + 1e-12
    noise_pwr = np.mean(no ** 2) + 1e-12

    target_noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_pwr / noise_pwr)
    mixed = s + no * scale

    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed *= 0.95 / peak
    return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    out_root:     str  = "audio/data/processed",
    n_train_pos:  int  = 300,
    n_train_neg:  int  = 300,
    n_val_pos:    int  = 80,
    n_val_neg:    int  = 80,
    duration_s:   float = 2.0,
    seed:         int  = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    root = Path(out_root)
    splits = {
        "train": {"positive": n_train_pos, "negative": n_train_neg},
        "val":   {"positive": n_val_pos,   "negative": n_val_neg},
    }

    siren_fns    = [siren_wail, yelp_siren, phaser_siren]
    noise_fns    = [road_noise, crowd_noise]
    snr_levels   = [-5, 0, 5, 10, 15, 20]

    total = sum(v for s in splits.values() for v in s.values())
    done  = 0

    for split, classes in splits.items():
        for cls, n_samples in classes.items():
            out_dir = root / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_samples):
                if cls == "positive":
                    # Pick a siren type and an SNR
                    gen   = random.choice(siren_fns)
                    noise = random.choice(noise_fns)(duration_s)
                    sig   = gen(duration_s)
                    snr   = random.choice(snr_levels)
                    wav   = mix_at_snr(sig, noise, snr)
                    name  = f"siren_{gen.__name__}_{i:04d}_snr{snr:+d}.wav"
                else:
                    # Pure noise -- no siren signal
                    gen  = random.choice(noise_fns)
                    wav  = gen(duration_s)
                    wav  = wav.astype(np.float32)
                    name = f"noise_{gen.__name__}_{i:04d}.wav"

                sf.write(out_dir / name, wav, SR)
                done += 1

                if done % 50 == 0 or done == total:
                    print(f"  [{done}/{total}] generated {split}/{cls}/{name}")

    print(f"\n[OK] Synthetic dataset written to: {root}")
    print(f"  Train: {n_train_pos} pos, {n_train_neg} neg")
    print(f"  Val:   {n_val_pos} pos, {n_val_neg} neg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic siren training data")
    ap.add_argument("--out",         default="audio/data/processed",
                    help="Output root directory (default: audio/data/processed)")
    ap.add_argument("--train-pos",   type=int, default=300, help="# train positive samples")
    ap.add_argument("--train-neg",   type=int, default=300, help="# train negative samples")
    ap.add_argument("--val-pos",     type=int, default=80,  help="# val positive samples")
    ap.add_argument("--val-neg",     type=int, default=80,  help="# val negative samples")
    ap.add_argument("--duration",    type=float, default=2.0, help="Clip duration (s)")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    print("=" * 60)
    print("Green Wave++ -- Synthetic Audio Generator")
    print("=" * 60)
    print(f"Generating {args.train_pos + args.train_neg + args.val_pos + args.val_neg} samples...\n")

    build_dataset(
        out_root    = args.out,
        n_train_pos = args.train_pos,
        n_train_neg = args.train_neg,
        n_val_pos   = args.val_pos,
        n_val_neg   = args.val_neg,
        duration_s  = args.duration,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
