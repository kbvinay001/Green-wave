"""Second corridor: the SUMO comparison is integration-only; here we lock the
network artifacts and the recorded summary so the generalization test stays
reproducible."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NET = ROOT / "sim" / "nets" / "shollinganallur"
RESULTS = ROOT / "evaluation" / "results" / "second_corridor.json"


def test_network_artifacts_exist():
    assert (NET / "shollinganallur.net.xml").exists()
    assert (NET / "background.rou.xml").exists()
    assert (NET / "ev.rou.xml").exists()
    corr = json.loads((NET / "_config_corridor.json").read_text())
    assert corr["lane_id"] and len(corr["intersections"]) >= 4
    assert all({"id", "distance_m", "approach_edge"} <= set(i) for i in corr["intersections"])


def test_shol_config_swaps_in_the_corridor():
    from evaluation.second_corridor import shol_config
    cfg = shol_config()
    corr = cfg["intersection"]["corridors"][0]
    assert corr["lane_id"] == "approach_main"
    # fusion params still come from the base config (network-independent)
    assert "fusion" in cfg and "sumo" in cfg


@pytest.mark.skipif(not RESULTS.exists(), reason="run `python -m evaluation.second_corridor` first")
def test_recorded_generalization_summary_is_well_formed():
    s = json.loads(RESULTS.read_text())["summary"]
    assert s["corridor"] == "shollinganallur_chennai"
    assert s["signals"] >= 4 and s["n"] >= 2
    # we record whatever the result is (this corridor is a hard, dense case);
    # the test just guards that the pipeline produced a real paired number
    assert "ev_time_saved_s" in s and "p_value" in s and s["ci95"]
