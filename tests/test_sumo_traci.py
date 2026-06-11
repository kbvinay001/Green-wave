"""
TraCI integration tests for fusion/sumo_controller.py.

Runs real SUMO headless on the generated 3-junction corridor
(sim/nets/test_corridor.*).  Skipped when SUMO is not installed.

Also serves as the task-9 verification: the green cascade must fire each
TLS at ITS OWN ETA in simulation time, not all simultaneously.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUMO_HOME = os.environ.get("SUMO_HOME", "")
SUMO_OK = bool(SUMO_HOME) and (Path(SUMO_HOME) / "bin").exists()
NET_OK = (ROOT / "sim" / "nets" / "test_corridor.sumocfg").exists()

pytestmark = pytest.mark.skipif(not (SUMO_OK and NET_OK),
                                reason="SUMO or test network not available")

CFG = {
    "sumo": {
        "step_length": 0.1,
        "all_red_duration": 2.5,
        "preempt_green_duration": 5.0,     # short hold keeps the test fast
        "downstream_lookahead": 3,
    },
    "intersection": {
        "corridors": [{
            "lane_id": "ev_corridor_we",
            "intersections": [
                {"id": "A0", "distance_m": 0,   "approach_edge": "left0A0"},
                {"id": "B0", "distance_m": 200, "approach_edge": "A0B0"},
                {"id": "C0", "distance_m": 400, "approach_edge": "B0C0"},
            ],
        }],
    },
}

ETAS = [3.0, 8.0, 13.0]


@pytest.fixture
def controller():
    from fusion.sumo_controller import SumoController
    ctrl = SumoController(CFG, sumo_cfg=str(ROOT / "sim" / "nets" / "test_corridor.sumocfg"))
    assert not ctrl._mock, "controller fell back to mock -- traci import failed?"
    ctrl.start()
    yield ctrl
    ctrl.stop()


def _approach_green(ctrl, tls_id: str) -> bool:
    return ctrl.get_tls_states().get(tls_id) == "green"


def test_cascade_fires_per_tls_at_eta(controller):
    ctrl = controller

    # Let the sim warm up so cross traffic owns some phases
    for _ in range(50):                          # 5.0 s
        ctrl.step()
    t0 = ctrl.sim_time()

    ctrl.trigger_preemption("ev_corridor_we", ["A0", "B0", "C0"], ETAS)

    first_green = {}                             # tls -> sim time it turned green
    while ctrl.sim_time() < t0 + 16.0:
        ctrl.step()
        for tls in ("A0", "B0", "C0"):
            if tls not in first_green and _approach_green(ctrl, tls):
                first_green[tls] = ctrl.sim_time() - t0

    assert set(first_green) == {"A0", "B0", "C0"}, \
        f"not every TLS turned green: {first_green}"

    # each TLS goes green at ITS eta (one step-length tolerance)
    for tls, eta in zip(("A0", "B0", "C0"), ETAS):
        assert abs(first_green[tls] - eta) <= 0.3, \
            f"{tls} green at t+{first_green[tls]:.1f}s, expected ~{eta}s"

    # staggered, not simultaneous
    times = [first_green[t] for t in ("A0", "B0", "C0")]
    assert times[0] < times[1] < times[2]
    assert times[2] - times[0] >= 9.0


def test_all_red_clearance_before_first_green(controller):
    ctrl = controller
    for _ in range(30):
        ctrl.step()
    t0 = ctrl.sim_time()
    ctrl.trigger_preemption("ev_corridor_we", ["A0", "B0", "C0"], ETAS)

    # during clearance (first 2.5s) every corridor TLS shows red on approach
    while ctrl.sim_time() < t0 + 2.0:
        ctrl.step()
        assert not any(_approach_green(ctrl, t) for t in ("A0", "B0", "C0"))


def test_program_restored_after_hold(controller):
    ctrl = controller
    import traci
    original = {t: traci.trafficlight.getProgram(t) for t in ("A0", "B0", "C0")}

    for _ in range(20):
        ctrl.step()
    t0 = ctrl.sim_time()
    ctrl.trigger_preemption("ev_corridor_we", ["A0", "B0", "C0"], ETAS)

    # run past last green (13) + hold (5) + margin
    while ctrl.sim_time() < t0 + 20.0:
        ctrl.step()

    for t in ("A0", "B0", "C0"):
        assert traci.trafficlight.getProgram(t) == original[t], \
            f"{t} program not restored"
    assert not ctrl.is_active("ev_corridor_we")


def test_duplicate_trigger_ignored(controller):
    ctrl = controller
    ctrl.step()
    ctrl.trigger_preemption("ev_corridor_we", ["A0", "B0", "C0"], ETAS)
    n_sched = len(ctrl._schedule)
    ctrl.trigger_preemption("ev_corridor_we", ["A0", "B0", "C0"], ETAS)
    assert len(ctrl._schedule) == n_sched      # second call added nothing
