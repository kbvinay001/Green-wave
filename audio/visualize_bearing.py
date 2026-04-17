#!/usr/bin/env python3
"""
Visualize bearing estimation on a synthetic moving source.
Saves:
  - outputs/bearing_curve.png (true vs estimated over time)
  - outputs/bearing_error_hist.png (error histogram)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bearing import MicrophoneArray, BearingEstimator

def simulate(sequence_seconds=2.0, sr=16000, hop_sec=0.1):
    positions = [[0.0, 0.0], [0.15, 0.0], [0.075, 0.13]]
    array = MicrophoneArray(positions)
    estimator = BearingEstimator(array, sample_rate=sr)

    chunk_samples = int(sr * hop_sec)
    n_chunks = int(sequence_seconds / hop_sec)
    t = np.linspace(0, hop_sec, chunk_samples, endpoint=False)

    true_angles, est_angles, confs, times = [], [], [], []

    for i in range(n_chunks):
        true_angle = (i / max(1, n_chunks - 1)) * 180.0  # sweep 0->180°
        source = np.sin(2 * np.pi * 1000 * t)            # 1 kHz tone

        channels = []
        angle_rad = np.radians(true_angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        for mic_pos in array.positions:
            delay = -np.dot(mic_pos, direction) / 343.0
            delay_samples = int(delay * sr)
            if delay_samples > 0:
                sig = np.pad(source, (delay_samples, 0))[:-delay_samples]
            elif delay_samples < 0:
                sig = np.pad(source, (0, -delay_samples))[-delay_samples:]
            else:
                sig = source.copy()
            # add a little noise
            sig += np.random.randn(len(sig)) * 0.03
            channels.append(sig)

        res = estimator.estimate_bearing(channels, use_median=True)
        true_angles.append(true_angle)
        est_angles.append(res["bearing_deg"])
        confs.append(res["confidence"])
        times.append(i * hop_sec)

    return np.array(times), np.array(true_angles), np.array(est_angles), np.array(confs)

def main():
    ROOT = Path(__file__).resolve().parents[1]
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    times, true_deg, est_deg, conf = simulate(sequence_seconds=2.0)

    # wrap error to [-180, 180], then absolute
    err = est_deg - true_deg
    err = (err + 180) % 360 - 180
    abs_err = np.abs(err)

    # Plot curves
    plt.figure(figsize=(10,4))
    plt.plot(times, true_deg, label="True")
    plt.plot(times, est_deg, label="Estimated")
    plt.xlabel("Time (s)")
    plt.ylabel("Bearing (deg)")
    plt.title("Bearing tracking (synthetic moving source)")
    plt.grid(True); plt.legend()
    curve_path = out_dir / "bearing_curve.png"
    plt.tight_layout(); plt.savefig(curve_path, dpi=150)
    print(f"[OK] Saved {curve_path}")

    # Error histogram
    plt.figure(figsize=(6,4))
    plt.hist(abs_err, bins=20)
    plt.xlabel("Absolute error (deg)")
    plt.ylabel("Count")
    plt.title(f"Error histogram (mean={abs_err.mean():.1f}°, median={np.median(abs_err):.1f}°)")
    plt.grid(True, axis="y")
    hist_path = out_dir / "bearing_error_hist.png"
    plt.tight_layout(); plt.savefig(hist_path, dpi=150)
    print(f"[OK] Saved {hist_path}")

    print(f"\nMean error: {abs_err.mean():.2f}° | Median: {np.median(abs_err):.2f}°")

if __name__ == "__main__":
    main()
