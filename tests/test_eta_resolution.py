"""Task 9: cascade ETAs must come from the route predictor's real distances."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.fuser import FusionCommand  # noqa: E402
from fusion.route_predictor import RoutePredictor  # noqa: E402
from integration.pipeline import resolve_command_etas  # noqa: E402

# Benz Circle geometry: junction spacing 0 / 158 / 384 / 831 m
BENZ_CFG = {
    "intersection": {
        "corridors": [{
            "lane_id": "approach_west",
            "heading_deg": 270,
            "intersections": [
                {"id": "tlsA", "distance_m": 0},
                {"id": "tlsB", "distance_m": 158},
                {"id": "tlsC", "distance_m": 384},
                {"id": "tlsD", "distance_m": 831},
            ],
        }],
    }
}


def make_cmd(**kw):
    base = dict(target_lane="approach_west",
                corridor_tls=["tlsA", "tlsB", "tlsC", "tlsD"],
                eta_seconds=[8.0, 11.0, 14.0, 17.0],   # the fuser's rough guess
                belief=0.9, timestamp=0.0)
    base.update(kw)
    return FusionCommand(**base)


def test_etas_use_real_corridor_distances():
    pred = RoutePredictor(config=BENZ_CFG)
    # ambulance 100 m before the first junction at 15 m/s
    cmd = make_cmd(speed_mps=15.0, distance_m=100.0)
    tls_ids, etas = resolve_command_etas(pred, cmd)

    assert tls_ids == ["tlsA", "tlsB", "tlsC", "tlsD"]
    assert etas == pytest.approx([100 / 15, 258 / 15, 484 / 15, 931 / 15], abs=0.1)
    # the far junction is ~a minute out -- nothing like the fuser's 17 s guess
    assert etas[-1] > 60


def test_no_speed_distance_keeps_fuser_guess():
    pred = RoutePredictor(config=BENZ_CFG)
    cmd = make_cmd(speed_mps=None, distance_m=None)
    tls_ids, etas = resolve_command_etas(pred, cmd)
    assert etas == [8.0, 11.0, 14.0, 17.0]


def test_unknown_lane_keeps_fuser_guess():
    pred = RoutePredictor(config=BENZ_CFG)
    cmd = make_cmd(target_lane="approach_mystery", speed_mps=15.0, distance_m=100.0)
    _, etas = resolve_command_etas(pred, cmd)
    assert etas == [8.0, 11.0, 14.0, 17.0]
