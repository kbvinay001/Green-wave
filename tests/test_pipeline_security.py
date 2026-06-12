"""The pipeline end of phase 4: rate-gated preemptions, all of it audited."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.audit import verify                      # noqa: E402
from fusion.fuser import FusionCommand               # noqa: E402
from integration.pipeline import EndToEndPipeline, load_config  # noqa: E402


def _demo_pipeline(tmp_path):
    cfg = load_config()
    cfg.setdefault("security", {})["audit_log"] = str(tmp_path / "audit.jsonl")
    return EndToEndPipeline(cfg, demo=True), tmp_path / "audit.jsonl"


def _cmd(ts):
    return FusionCommand(
        target_lane  = "approach_north",
        corridor_tls = ["J_N1", "J_N2"],
        eta_seconds  = [5.0, 11.0],
        belief       = 0.9,
        timestamp    = ts,
        speed_mps    = 16.0,
        distance_m   = 80.0,
    )


def test_fifth_preemption_in_the_hour_is_blocked(tmp_path):
    pipe, audit_path = _demo_pipeline(tmp_path)

    results = [pipe._handle_preemption(_cmd(t), ts=t) for t in (0, 600, 1200, 1800, 2400, 3000)]
    assert results == [True, True, True, True, False, False]

    # ...but the window slides: an hour after the first one, a slot frees
    assert pipe._handle_preemption(_cmd(3601.0), ts=3601.0) is True


def test_other_lanes_keep_their_own_budget(tmp_path):
    pipe, _ = _demo_pipeline(tmp_path)
    for t in range(4):
        assert pipe._handle_preemption(_cmd(float(t)), ts=float(t))
    assert not pipe._handle_preemption(_cmd(4.0), ts=4.0)

    east = FusionCommand("approach_east", ["J_E1"], [4.0], 0.85, 5.0,
                         speed_mps=14.0, distance_m=60.0)
    assert pipe._handle_preemption(east, ts=5.0)


def test_every_decision_lands_in_the_audit_chain(tmp_path):
    pipe, audit_path = _demo_pipeline(tmp_path)
    for t in (0, 1, 2, 3, 4):     # 4 fire, 1 denied
        pipe._handle_preemption(_cmd(float(t)), ts=float(t))

    ok, bad = verify(audit_path)
    assert ok, f"audit chain broken at seq {bad}"

    events = [json.loads(l)["event"] for l in audit_path.read_text().splitlines()]
    assert events == (["pipeline_start"]
                      + ["preempt_fired"] * 4
                      + ["preempt_denied_rate_limit"])

    denied = json.loads(audit_path.read_text().splitlines()[-1])
    assert denied["payload"]["lane"] == "approach_north"
    assert denied["payload"]["retry_after_sec"] > 0
