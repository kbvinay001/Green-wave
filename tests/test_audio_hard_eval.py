"""The siren hard-test rests on SNR mixing being correct -- so verify it."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audio"))

from preprocess import AudioAugmenter  # noqa: E402


def _measured_snr_db(signal, noise):
    return 10.0 * np.log10(np.mean(signal ** 2) / np.mean(noise ** 2))


@pytest.mark.parametrize("target", [-5.0, 0.0, 5.0, 10.0])
def test_set_snr_hits_the_target(target):
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(48000).astype(np.float32) * 0.1
    noise = rng.standard_normal(48000).astype(np.float32)
    aug = AudioAugmenter(sample_rate=16000)
    _, noise_adj = aug.set_snr(sig, noise, target)
    # noise_adjusted is scaled to exactly hit the requested ratio (clipping
    # only touches the returned mix, not this measurement)
    assert _measured_snr_db(sig, noise_adj) == pytest.approx(target, abs=0.1)


def test_set_snr_prevents_clipping():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(16000).astype(np.float32)      # already large
    noise = rng.standard_normal(16000).astype(np.float32)
    mixed, _ = AudioAugmenter(16000).set_snr(sig, noise, -5.0)
    assert np.abs(mixed).max() <= 0.95 + 1e-6


def test_lower_snr_means_more_noise_energy():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(16000).astype(np.float32) * 0.1
    noise = rng.standard_normal(16000).astype(np.float32)
    aug = AudioAugmenter(16000)
    _, n_loud = aug.set_snr(sig, noise, -5.0)   # far siren -> louder noise
    _, n_quiet = aug.set_snr(sig, noise, 10.0)  # near siren -> quieter noise
    assert np.mean(n_loud ** 2) > np.mean(n_quiet ** 2)


def test_random_segment_crops_and_pads():
    aug = AudioAugmenter(16000)
    long = np.ones(50000, dtype=np.float32)
    assert aug.random_segment(long, 48000).shape == (48000,)
    short = np.ones(1000, dtype=np.float32)
    out = aug.random_segment(short, 48000)
    assert out.shape == (48000,)
    assert out[:1000].sum() == 1000 and out[1000:].sum() == 0   # zero-padded tail
