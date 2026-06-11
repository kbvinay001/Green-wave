"""Cross-modal gate: a siren alone must never fire a full preemption."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.fuser import Lane, TemporalFusionEngine  # noqa: E402

CFG = {
    "fusion": {
        "decay_factor": 0.92,
        "sigma_angle_deg": 20.0,
        "arm_threshold": 0.6,
        "arm_duration_sec": 0.5,
        "preempt_threshold": 0.8,
        "preempt_threshold_approaching": 0.7,
        "min_visual_speed_kmh": 30.0,
        "audio_only_belief_cap": 0.7,
        "vision_confirm_window_sec": 3.0,
    }
}

VISION_DET = {
    "lane_id": "approach_north", "confidence": 0.9, "approaching": True,
    "distance_m": 120.0, "speed_kmh": 55.0, "speed_mps": 55.0 / 3.6,
}


def make_engine():
    return TemporalFusionEngine([Lane("approach_north", 0.0, ["J1", "J2"])], CFG)


def test_audio_alone_caps_at_07_and_never_preempts():
    eng = make_engine()
    fired = []
    # 8 seconds of a loud, perfectly-aligned siren and nothing else
    for tick in range(80):
        fired += eng.update(audio_conf=1.0, audio_bearing=0.0,
                            vision_detections=[], timestamp=tick * 0.1)
    assert eng.get_beliefs()["approach_north"] <= 0.7 + 1e-9
    assert fired == [], "audio-only input must not trigger preemption"
    # it does ARM though -- that's what earns the graded green extension
    assert eng.get_phases()["approach_north"] == "ARMED"


def test_vision_confirmation_unlocks_preemption():
    eng = make_engine()
    fired = []
    t = 0.0
    for tick in range(30):                       # siren first
        t = tick * 0.1
        fired += eng.update(1.0, 0.0, [], t)
    assert fired == []
    for tick in range(30, 60):                   # then the camera sees it too
        t = tick * 0.1
        fired += eng.update(1.0, 0.0, [dict(VISION_DET)], t)
    assert len(fired) == 1, "vision confirmation should release the gate"
    assert fired[0].belief >= 0.8


def test_confirmation_expires_after_window():
    # White-box: plant a high belief with a vision stamp at t=0, then watch
    # only decay + the cap. While the confirmation is fresh the belief may sit
    # above 0.7; one tick after the window closes it must be clamped back.
    eng = make_engine()
    eng.update(0.0, None, [dict(VISION_DET)], 0.0)     # stamps last_vision_time
    eng.states["approach_north"].belief = 0.95

    eng.update(0.0, None, [], 2.9)                     # window still open
    b_inside = eng.get_beliefs()["approach_north"]
    assert b_inside > 0.7                              # decayed 0.95, not clamped

    eng.update(0.0, None, [], 3.2)                     # window closed
    assert eng.get_beliefs()["approach_north"] <= 0.7 + 1e-9


def test_vision_confirmed_helper():
    eng = make_engine()
    assert not eng.vision_confirmed("approach_north", 0.0)
    eng.update(0.0, None, [dict(VISION_DET)], 1.0)
    assert eng.vision_confirmed("approach_north", 2.0)
    assert not eng.vision_confirmed("approach_north", 5.0)
