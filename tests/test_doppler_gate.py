"""Doppler gate: receding sirens (falling pitch trend) get discounted."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audio"))

from stream_detector import DopplerTracker  # noqa: E402

SR = 16000


def tone_with_freq_curve(freq_at, duration, sr=SR, amp=0.3):
    """Synthesize a tone whose instantaneous frequency follows freq_at(t)."""
    t = np.arange(int(duration * sr)) / sr
    phase = 2 * np.pi * np.cumsum(freq_at(t)) / sr
    return (amp * np.sin(phase)).astype(np.float32)


def feed(tracker, audio, chunk=1600):
    out = {"doppler_factor": 1.0, "pitch_slope_hz_s": 0.0}
    t = 0.0
    for i in range(0, len(audio) - chunk + 1, chunk):
        t += chunk / SR
        out = tracker.update(audio[i:i + chunk], t)
    return out


def make_tracker():
    return DopplerTracker(sample_rate=SR, window_sec=2.0, block_sec=0.5,
                          band_hz=(500, 2500),
                          falling_slope_hz_per_s=-10.0, receding_factor=0.3)


def test_steady_tone_is_not_receding():
    audio = tone_with_freq_curve(lambda t: np.full_like(t, 1200.0), 3.0)
    out = feed(make_tracker(), audio)
    assert out["doppler_factor"] == 1.0
    assert abs(out["pitch_slope_hz_s"]) < 10


def test_falling_pitch_is_receding():
    # 1300 -> 1100 Hz over 3 s = -67 Hz/s, a vehicle clearly driving away
    audio = tone_with_freq_curve(lambda t: 1300.0 - 67.0 * t, 3.0)
    out = feed(make_tracker(), audio)
    assert out["doppler_factor"] == 0.3
    assert out["pitch_slope_hz_s"] < -10


def test_rising_pitch_is_approaching():
    audio = tone_with_freq_curve(lambda t: 1100.0 + 50.0 * t, 3.0)
    out = feed(make_tracker(), audio)
    assert out["doppler_factor"] == 1.0


def test_wail_sweep_without_drift_is_not_receding():
    # The killer case: a wail sweeps 800-1400 Hz at 0.4 Hz. Raw pitch slope
    # oscillates by hundreds of Hz/s, but the sweep's top stays put, so the
    # envelope trend must say "not receding".
    sweep = lambda t: 1100.0 + 300.0 * np.sin(2 * np.pi * 0.4 * t)
    audio = tone_with_freq_curve(sweep, 4.0)
    out = feed(make_tracker(), audio)
    assert out["doppler_factor"] == 1.0


def test_too_little_history_stays_neutral():
    tracker = make_tracker()
    audio = tone_with_freq_curve(lambda t: 1300.0 - 67.0 * t, 0.6)
    out = feed(tracker, audio)
    assert out["doppler_factor"] == 1.0   # can't call a trend from <1 s


def test_reset_clears_history():
    tracker = make_tracker()
    feed(tracker, tone_with_freq_curve(lambda t: 1300.0 - 67.0 * t, 3.0))
    tracker.reset()
    assert tracker.verdict()["doppler_factor"] == 1.0
