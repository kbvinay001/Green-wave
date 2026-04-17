#!/usr/bin/env python3
"""
Quick synthetic dataset: simple wail/yelp sirens + noise beds.
This is just to bootstrap the pipeline; replace with real data later.
"""
import numpy as np
from pathlib import Path
import soundfile as sf
rng = np.random.default_rng(123)

SR = 16000

def tone(f, dur, sr=SR):
    t = np.arange(int(dur*sr)) / sr
    return np.sin(2*np.pi*f*t).astype(np.float32)

def chirp(f0, f1, dur, sr=SR):
    t = np.arange(int(dur*sr)) / sr
    k = (f1-f0)/dur
    phase = 2*np.pi*(f0*t + 0.5*k*t**2)
    return np.sin(phase).astype(np.float32)

def wail(dur=2.0):
    # slow up-down between 500–1000 Hz
    half = dur/2
    up = chirp(500,1000,half)
    down = chirp(1000,500,half)
    s = np.concatenate([up,down])
    env = np.hanning(len(s))
    return (s*env).astype(np.float32)

def yelp(dur=2.0):
    # faster 700–1600 Hz sweeps
    segs = []
    t = 0.0
    while t < dur:
        seg = chirp(700,1600,0.25)
        segs.append(seg)
        t += 0.25
    s = np.concatenate(segs)[:int(dur*SR)]
    env = np.hanning(len(s))
    return (s*env).astype(np.float32)

def noise_bed(dur=2.0):
    # pinkish noise + a few honks/rumble hints
    n = rng.normal(0, 0.05, int(dur*SR)).astype(np.float32)
    # low-frequency rumble
    n += 0.03*tone(80, dur)
    # occasional honk blips
    for _ in range(rng.integers(1,3)):
        start = rng.integers(0, int((dur-0.2)*SR))
        n[start:start+int(0.2*SR)] += 0.2*tone(440, 0.2)[:int(0.2*SR)]
    # normalize
    n /= max(1e-6, np.max(np.abs(n)))
    return n*0.5

def mix_at_snr(s, n, snr_db):
    s_rms = np.sqrt(np.mean(s**2) + 1e-12)
    n_rms = np.sqrt(np.mean(n**2) + 1e-12)
    tgt_n_rms = s_rms / (10**(snr_db/20))
    n = n * (tgt_n_rms/(n_rms+1e-12))
    y = s + n
    y = y / max(1.0, np.max(np.abs(y))/0.95)
    return y

def main():
    root = Path(__file__).resolve().parents[1]
    sir_dir = root/"data"/"raw"/"sirens"
    noi_dir = root/"data"/"raw"/"noise"
    sir_dir.mkdir(parents=True, exist_ok=True)
    noi_dir.mkdir(parents=True, exist_ok=True)

    # Make 50 positives (wail/yelp mixes) and 50 negatives (noise)
    idx = 0
    for i in range(25):
        s = wail(2.0); n = noise_bed(2.0)
        y = mix_at_snr(s,n, rng.choice([-5,0,5,10,15]))
        sf.write(sir_dir/f"synth_wail_{i:03d}.wav", y, SR, subtype="PCM_16")
    for i in range(25):
        s = yelp(2.0); n = noise_bed(2.0)
        y = mix_at_snr(s,n, rng.choice([-5,0,5,10,15]))
        sf.write(sir_dir/f"synth_yelp_{i:03d}.wav", y, SR, subtype="PCM_16")
    for i in range(50):
        n = noise_bed(2.0)
        sf.write(noi_dir/f"synth_noise_{i:03d}.wav", n, SR, subtype="PCM_16")
    print("[OK] Wrote synthetic WAVs to:", sir_dir, "and", noi_dir)

if __name__ == "__main__":
    main()
