"""Graded action: arming a lane stretches the nearest signal's green by 5s."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.fuser import Lane, TemporalFusionEngine  # noqa: E402
from fusion.sumo_controller import SumoController, TLSPhase  # noqa: E402

CFG = {
    "fusion": {
        "decay_factor": 0.92, "sigma_angle_deg": 20.0,
        "arm_threshold": 0.6, "arm_duration_sec": 0.5,
        "preempt_threshold": 0.8, "min_visual_speed_kmh": 30.0,
        "audio_only_belief_cap": 0.7, "vision_confirm_window_sec": 3.0,
        "arm_green_extension_sec": 5.0,
    },
    "sumo": {
        "step_length": 0.1, "all_red_duration": 2.5,
        "preempt_green_duration": 5.0, "downstream_lookahead": 3,
    },
}


def test_arm_event_fires_once_per_arm_cycle():
    eng = TemporalFusionEngine([Lane("approach_north", 0.0, ["J1", "J2"])], CFG)

    t = 0.0
    for tick in range(40):                      # siren pushes belief past 0.6
        t = tick * 0.1
        eng.update(1.0, 0.0, [], t)
    assert eng.pop_arm_events() == ["approach_north"]
    assert eng.pop_arm_events() == []           # drained

    for tick in range(40, 60):                  # staying armed adds nothing
        eng.update(1.0, 0.0, [], tick * 0.1)
    assert eng.pop_arm_events() == []

    # silence -> belief falls -> disarm -> new siren -> a second arm event
    t = 6.0
    while eng.get_beliefs()["approach_north"] >= 0.05:
        t += 0.5
        eng.update(0.0, None, [], t)
    for tick in range(40):
        t += 0.1
        eng.update(1.0, 0.0, [], t)
    assert eng.pop_arm_events() == ["approach_north"]


def test_mock_extend_green_only_when_green():
    ctrl = SumoController(CFG, mock=True)
    ctrl._mock_ctrl.set_phase("J_N1", TLSPhase.GREEN)
    assert ctrl.extend_green("J_N1", 5.0) is True
    ctrl._mock_ctrl.set_phase("J_N1", TLSPhase.RED)
    assert ctrl.extend_green("J_N1", 5.0) is False


# ---------------------------------------------------------------------------
# Real SUMO: extension visibly moves the phase switch time
# ---------------------------------------------------------------------------

SUMO_OK = bool(os.environ.get("SUMO_HOME"))
NET_OK = (ROOT / "sim" / "nets" / "test_corridor.sumocfg").exists()


@pytest.mark.skipif(not (SUMO_OK and NET_OK), reason="SUMO not available")
def test_traci_extend_green_moves_next_switch():
    import traci

    cfg = dict(CFG)
    cfg["intersection"] = {"corridors": [{
        "lane_id": "ev",
        "intersections": [
            {"id": "A0", "distance_m": 0, "approach_edge": "left0A0"},
        ],
    }]}
    ctrl = SumoController(cfg, sumo_cfg=str(ROOT / "sim" / "nets" / "test_corridor.sumocfg"))
    ctrl.start()
    try:
        # step until the A0 approach happens to be green
        for _ in range(600):
            ctrl.step()
            if ctrl.get_tls_states().get("A0") == "green":
                break
        assert ctrl.get_tls_states()["A0"] == "green", "approach never went green"

        before = traci.trafficlight.getNextSwitch("A0")
        assert ctrl.extend_green("A0", 5.0) is True
        after = traci.trafficlight.getNextSwitch("A0")
        assert after - before == pytest.approx(5.0, abs=0.2)

        # and when the approach is red, nothing happens
        for _ in range(1200):
            ctrl.step()
            if ctrl.get_tls_states().get("A0") == "red":
                break
        if ctrl.get_tls_states()["A0"] == "red":
            assert ctrl.extend_green("A0", 5.0) is False
    finally:
        ctrl.stop()
