"""Tests for integration/replay.py virtual sensor sources (no models needed)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.replay import (  # noqa: E402
    AudioDetectorAdapter,
    AudioFileSource,
    VideoFileSource,
    VirtualClock,
    VirtualSensorRig,
    VisionDetectorAdapter,
)

SR = 16000


# ---------------------------------------------------------------------------
# Fixtures: tiny media files
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wav_mono(tmp_path_factory):
    sf = pytest.importorskip("soundfile")
    p = tmp_path_factory.mktemp("media") / "tone_mono.wav"
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * 800 * t)).astype(np.float32), SR)
    return p


@pytest.fixture(scope="module")
def wav_3ch(tmp_path_factory):
    sf = pytest.importorskip("soundfile")
    p = tmp_path_factory.mktemp("media") / "tone_3ch.wav"
    t = np.linspace(0, 1.0, SR, endpoint=False)
    x = 0.3 * np.sin(2 * np.pi * 700 * t).astype(np.float32)
    sf.write(str(p), np.stack([x, x * 0.9, x * 0.8], axis=1), SR)
    return p


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    p = tmp_path_factory.mktemp("media") / "clip.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert w.isOpened()
    for i in range(20):                       # 2.0 s @ 10 fps
        frame = np.full((48, 64, 3), i * 10 % 255, np.uint8)
        w.write(frame)
    w.release()
    return p


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def test_clock_speed_mapping():
    clk = VirtualClock(speed=2.0)
    clk.start()
    time.sleep(0.2)
    assert 0.3 < clk.media_now() < 0.7        # ~0.4 media seconds

def test_clock_sleep_until_paces():
    clk = VirtualClock(speed=1.0)
    clk.start()
    t0 = time.time()
    clk.sleep_until(0.25)
    assert 0.2 <= time.time() - t0 <= 0.5

def test_clock_sleep_until_past_is_noop():
    clk = VirtualClock()
    clk.start()
    time.sleep(0.05)
    t0 = time.time()
    clk.sleep_until(0.01)                     # already past
    assert time.time() - t0 < 0.05


# ---------------------------------------------------------------------------
# File sources
# ---------------------------------------------------------------------------

def test_audio_source_mono_chunks(wav_mono):
    src = AudioFileSource(str(wav_mono), sr=SR, chunk_sec=0.1)
    items = list(src)
    assert len(items) == 20                   # 2.0 s / 0.1 s
    channels, t = items[0]
    assert len(channels) == 1 and len(channels[0]) == 1600
    ts = [t for _, t in items]
    assert ts[0] == 0.0
    assert all(abs((b - a) - 0.1) < 1e-6 for a, b in zip(ts, ts[1:]))

def test_audio_source_multichannel(wav_3ch):
    src = AudioFileSource(str(wav_3ch), sr=SR, chunk_sec=0.1)
    assert src.n_channels == 3
    channels, _ = next(iter(src))
    assert len(channels) == 3

def test_video_source_frames(video_file):
    src = VideoFileSource(str(video_file))
    items = list(src)
    src.release()
    assert len(items) == 20
    frame, t0 = items[0]
    assert frame.shape == (48, 64, 3)
    assert abs(items[1][1] - items[0][1] - 0.1) < 1e-6   # 10 fps spacing


# ---------------------------------------------------------------------------
# Adapters (stubs -- no models)
# ---------------------------------------------------------------------------

def test_vision_adapter_maps_to_fusion_dict():
    stub = lambda frame, t: [{"confidence": 0.9}]        # no speed, no approaching
    ad = VisionDetectorAdapter("approach_east", assume_approaching=True,
                               default_speed_kmh=42.0, detector=stub)
    out = ad.process(np.zeros((48, 64, 3), np.uint8), 1.0)
    assert out == [{
        "lane_id": "approach_east", "confidence": 0.9, "approaching": True,
        "distance_m": None, "speed_kmh": 42.0, "speed_mps": pytest.approx(42.0 / 3.6),
    }]

def test_vision_adapter_prefers_tracker_values():
    stub = lambda frame, t: [{"confidence": 0.8, "approaching": False, "speed_kmh": 61.0}]
    ad = VisionDetectorAdapter("approach_east", detector=stub)
    out = ad.process(np.zeros((4, 4, 3), np.uint8), 0.0)
    assert out[0]["approaching"] is False
    assert out[0]["speed_kmh"] == 61.0

def test_audio_adapter_stub_passthrough():
    stub = lambda ch: {"p_siren": 0.7, "detected": True,
                       "bearing_deg": 12.0, "bearing_confidence": 0.5}
    ad = AudioDetectorAdapter(detector=stub)
    assert ad.process([np.zeros(1600)])["p_siren"] == 0.7


# ---------------------------------------------------------------------------
# Rig synchronization
# ---------------------------------------------------------------------------

def test_rig_synchronizes_audio_and_video(wav_mono, video_file):
    """Items with the same media timestamp must arrive close in wall time."""
    audio_src = AudioFileSource(str(wav_mono), sr=SR, chunk_sec=0.1)
    video_src = VideoFileSource(str(video_file))
    a_stub = lambda ch: {"p_siren": 0.0, "detected": False,
                         "bearing_deg": 0.0, "bearing_confidence": 0.0}
    v_stub = lambda frame, t: []

    rig = VirtualSensorRig(
        audio_src, video_src,
        AudioDetectorAdapter(detector=a_stub),
        VisionDetectorAdapter("approach_north", detector=v_stub),
        speed=4.0,                             # 2 s media in ~0.5 s wall
        on_finished=lambda n: None,
    )

    audio_events, video_events = [], []
    rig.start(
        put_audio=lambda item: audio_events.append((item[0]["t_media"], time.time())),
        put_vision=lambda item: video_events.append(item[1]),
    )
    rig.join(timeout=10)
    assert rig.finished

    assert len(audio_events) == 20
    assert len(video_events) == 20

    # pacing: 2.0 media seconds at 4x => ~0.5 s wall between first and last
    wall_span = audio_events[-1][1] - audio_events[0][1]
    assert 0.3 < wall_span < 1.5

    # sync: media t=1.0 lands at the same wall moment on both streams (+-0.25s)
    a_wall_at_1s = next(w for t, w in audio_events if abs(t - 1.0) < 0.01)
    v_wall_at_1s = video_events[10]            # frame 10 @ 10fps = t_media 1.0
    assert abs(a_wall_at_1s - v_wall_at_1s) < 0.25

def test_rig_requires_a_source():
    with pytest.raises(ValueError):
        VirtualSensorRig(None, None, None, None)


# ---------------------------------------------------------------------------
# Real-model integration (runs only where the trained checkpoint exists)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (ROOT / "checkpoints" / "audio_best.pt").exists(),
                    reason="trained audio checkpoint not present")
def test_audio_adapter_real_model_mono_bearing(wav_mono):
    ad = AudioDetectorAdapter(fallback_bearing_deg=37.0)
    src = AudioFileSource(str(wav_mono), sr=SR, chunk_sec=0.1)
    results = [ad.process(ch) for ch, _ in src]
    assert all(set(r) >= {"p_siren", "detected", "bearing_deg"} for r in results)
    # a plain 800 Hz tone may or may not trip the CRNN; if it does, the
    # fallback bearing must be reported
    for r in results:
        if r["detected"]:
            assert r["bearing_deg"] == 37.0
