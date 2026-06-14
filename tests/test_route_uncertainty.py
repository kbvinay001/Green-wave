"""Route-uncertainty: the deviation result is integration-tested by the SUMO
run; here we lock the recorded summary invariants so a regression is visible."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.route_uncertainty import DEVIATION_EDGE  # noqa: E402

RESULTS = ROOT / "evaluation" / "results" / "route_uncertainty.json"


def test_deviation_edge_is_configured():
    assert DEVIATION_EDGE and isinstance(DEVIATION_EDGE, str)


@pytest.mark.skipif(not RESULTS.exists(), reason="run `python -m evaluation.route_uncertainty` first")
def test_recorded_run_holds_the_safety_invariant():
    data = json.loads(RESULTS.read_text())
    s = data["summary"]
    # the whole point: a wrong route never strands the network on green
    assert s["all_preemptions_released"] is True
    # and the cost of being wrong is small and bounded
    assert s["max_extra_civilian_loss_s"] < 10.0
    # and the deviation actually happened in every seed (else we tested nothing)
    took, total = s["deviation_took_effect_in"].split("/")
    assert took == total


@pytest.mark.skipif(not RESULTS.exists(), reason="needs a recorded run")
def test_every_seed_released_and_finished():
    data = json.loads(RESULTS.read_text())
    for rec in data["rows"]:
        for tag in ("ctl", "dev"):
            assert rec[tag]["released_by_end"] is True
