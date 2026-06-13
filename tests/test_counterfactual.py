"""Counterfactual eval: the pure parts -- parsing, stop counting, summary math."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.counterfactual import (  # noqa: E402
    EV_ID, EV_ROUTE, EV_TYPE, SyntheticSensors, civilian_summary,
    parse_tripinfo, stops_from_trace, summarize, write_eval_routes,
    write_run_cfg,
)

TRIPINFO = """<tripinfos>
    <tripinfo id="bg_1" depart="0.0" duration="120.5" timeLoss="33.2"
              waitingTime="21.0" departDelay="0.0"/>
    <tripinfo id="{ev}" depart="60.0" duration="95.0" timeLoss="40.1"
              waitingTime="12.5" departDelay="1.2"/>
    <tripinfo id="bg_2" depart="5.0" duration="200.0" timeLoss="80.4"
              waitingTime="60.2" departDelay="3.0"/>
</tripinfos>"""


def test_parse_tripinfo_separates_ev_from_civilians(tmp_path):
    p = tmp_path / "trip.xml"
    p.write_text(TRIPINFO.format(ev=EV_ID))
    ev, civilians = parse_tripinfo(p)
    assert ev["duration"] == 95.0 and ev["departDelay"] == 1.2
    assert sorted(c["id"] for c in civilians) == ["bg_1", "bg_2"]


def test_parse_tripinfo_missing_ev(tmp_path):
    p = tmp_path / "trip.xml"
    p.write_text(TRIPINFO.format(ev="someone_else"))
    ev, civilians = parse_tripinfo(p)
    assert ev is None and len(civilians) == 3


def test_stops_from_trace():
    # drive, stop 3 ticks, drive, stop 2 ticks -> 2 stops, 0.5s standing
    trace = [(0.0, 10.0), (0.1, 5.0), (0.2, 0.1), (0.3, 0.0), (0.4, 0.2),
             (0.5, 4.0), (0.6, 9.0), (0.7, 0.1), (0.8, 0.0), (0.9, 8.0)]
    stops, stopped_s = stops_from_trace(trace)
    assert stops == 2
    assert stopped_s == pytest.approx(0.5, abs=0.01)


def test_stops_from_trace_never_stopped():
    assert stops_from_trace([(0.0, 8.0), (0.1, 9.0), (0.2, 12.0)]) == (0, 0.0)


def test_civilian_summary():
    s = civilian_summary([
        {"timeLoss": 10.0, "waitingTime": 4.0},
        {"timeLoss": 30.0, "waitingTime": 16.0},
    ])
    assert s == {"arrived": 2, "mean_time_loss_s": 20.0,
                 "max_time_loss_s": 30.0, "mean_waiting_s": 10.0}
    assert civilian_summary([])["arrived"] == 0


def test_write_run_cfg_is_valid_and_absolute(tmp_path):
    cfg_path, trip_path = write_run_cfg(tmp_path, "seed7_baseline", seed=7, scale=2.5)
    root = ET.parse(cfg_path).getroot()
    net = root.find("input/net-file").get("value")
    assert Path(net).is_absolute() and net.endswith("benz.net.xml")
    assert root.find("random_number/seed").get("value") == "7"
    assert root.find("processing/scale").get("value") == "2.5"
    assert root.find("output/tripinfo-output").get("value") == str(trip_path.resolve())


def test_eval_ev_has_no_bluelight(tmp_path):
    # same route as the live ambulance, but it must obey the signals
    path = write_eval_routes(tmp_path)
    root = ET.parse(path).getroot()
    vtype = root.find(f"vType[@id='{EV_TYPE}']")
    assert vtype is not None
    assert "has.bluelight.device" not in path.read_text()
    assert vtype.find("param") is None        # no devices of any kind
    edges = root.find(f"route[@id='{EV_ROUTE}']").get("edges")
    assert edges and "279067605#8" in edges        # runs the real corridor


def test_synthetic_sensors_respect_ranges():
    s = SyntheticSensors(seed=1, lane_id="approach_west", lane_heading_deg=270.0)
    # far away: silence, always
    for _ in range(50):
        assert s.audio(500.0) == (0.0, None)
        assert s.vision(500.0, 12.0) == []
    # vision range is tighter than audio range
    assert s.vision(120.0, 12.0) == []


def test_synthetic_sensors_are_seeded():
    a = [SyntheticSensors(7, "approach_west", 270.0).audio(100.0) for _ in range(1)]
    b = [SyntheticSensors(7, "approach_west", 270.0).audio(100.0) for _ in range(1)]
    assert a == b


def test_synthetic_sensors_hit_rates_and_noise():
    s = SyntheticSensors(seed=3, lane_id="approach_west", lane_heading_deg=270.0)
    hits, bearings = 0, []
    for _ in range(2000):
        conf, bearing = s.audio(100.0)
        if conf > 0.0:
            hits += 1
            assert 0.0 <= conf <= 1.0
            bearings.append(bearing)
    assert 0.80 <= hits / 2000 <= 0.90          # nominal 85%
    # bearing scatters around the lane heading, not uniformly
    assert all(abs((b - 270.0 + 180) % 360 - 180) < 25 for b in bearings)

    seen = 0
    for _ in range(2000):
        dets = s.vision(50.0, 14.0)
        if dets:
            seen += 1
            d = dets[0]
            assert d["lane_id"] == "approach_west" and d["approaching"]
            assert d["distance_m"] > 0 and 0.0 <= d["confidence"] <= 1.0
    assert 0.85 <= seen / 2000 <= 0.95          # nominal 90%


def test_summarize_other_mode_key_naming():
    mk = lambda t: {"ev_travel_s": t, "ev_time_loss_s": 0.0, "ev_stops": 1,
                    "ev_stopped_s": 0.0, "triggered_at": None,
                    "civilians": {"arrived": 5, "mean_time_loss_s": 10.0,
                                  "max_time_loss_s": 20.0, "mean_waiting_s": 0.0}}
    pairs = [{"seed": 1, "baseline": mk(100.0), "closedloop": mk(80.0)}]
    s = summarize(pairs, mode="closedloop")
    assert s["ev_travel_closedloop_s"]["mean"] == 80.0
    assert s["ev_time_saved_s"]["mean"] == 20.0


def test_summarize_math():
    def pair(seed, base_t, gw_t, civ_b, civ_g):
        mk = lambda t, civ, stops: {
            "ev_travel_s": t, "ev_time_loss_s": 0.0, "ev_stops": stops,
            "ev_stopped_s": 0.0, "triggered_at": None,
            "civilians": {"arrived": 10, "mean_time_loss_s": civ,
                          "max_time_loss_s": civ * 2, "mean_waiting_s": 0.0}}
        return {"seed": seed, "baseline": mk(base_t, civ_b, 3),
                "greenwave": mk(gw_t, civ_g, 0)}

    s = summarize([pair(1, 100.0, 60.0, 20.0, 24.0),
                   pair(2, 120.0, 70.0, 22.0, 28.0)])
    assert s["ev_travel_baseline_s"]["mean"] == 110.0
    assert s["ev_time_saved_s"]["mean"] == 45.0
    assert s["ev_time_saved_pct"] == pytest.approx(40.9, abs=0.1)
    assert s["civilian_extra_loss_s"]["mean"] == 5.0
    assert s["ev_stops_greenwave"]["mean"] == 0.0
