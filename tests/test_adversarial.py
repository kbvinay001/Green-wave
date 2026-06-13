"""
Adversarial / false-preemption -- the HARD CI GATE.

This drives the REAL TemporalFusionEngine. If any benign adversarial scenario
ever produces a single false preemption, the build fails here. That is the
point: the three trust gates (cross-modal cap, Doppler, rate limit) are only
worth claiming if something keeps them honest.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.adversarial import (  # noqa: E402
    AudioDet, NaiveFusionAdapter, RealFusionAdapter, ReferenceFusionAdapter,
    ScenarioResult, SCENARIOS, VisionDet, evaluate, load_config, run_all,
    run_comparison, run_scenario,
)

CFG = load_config()
GATE_SEEDS = 12   # enough to be meaningful, fast enough for CI


@pytest.fixture(scope="module")
def real_results():
    return run_all(CFG, seeds=GATE_SEEDS, use_real=True)


# ----------------------------------------------------------------------
# THE GATE
# ----------------------------------------------------------------------

def test_benign_scenarios_never_false_fire(real_results):
    """Zero false preemptions across every benign scenario. Hard gate."""
    benign = [r for r in real_results if r.expected == 0]
    leaks = {r.name: r.total_fires for r in benign if r.total_fires}
    assert not leaks, f"false preemption(s) leaked through a trust gate: {leaks}"


def test_spoof_flood_is_bounded(real_results):
    """A perfect dual-modal spoof WILL fire -- but stays within the cap."""
    spoof = next(r for r in real_results if r.name == "spoof_flood")
    cap = CFG["max_preempts_per_lane"]
    assert spoof.max_fires_in_any_seed >= 1          # it genuinely fires
    assert spoof.max_fires_in_any_seed <= cap        # ...but bounded


def test_overall_verdict_passes(real_results):
    ok, msgs = evaluate(real_results, CFG)
    assert ok, "; ".join(msgs)


# ----------------------------------------------------------------------
# The specific gates, isolated
# ----------------------------------------------------------------------

def test_cross_modal_cap_holds_on_audio_only(real_results):
    """Audio with no camera can arm but must never exceed the 0.7 cap."""
    phone = next(r for r in real_results if r.name == "phone_speaker")
    assert phone.peak_belief <= CFG["audio_cap"] + 1e-6


def test_cross_street_siren_barely_registers(real_results):
    """A ~90 deg-off siren should be almost entirely rejected on the corridor."""
    cross = next(r for r in real_results if r.name == "cross_street")
    assert cross.peak_belief < 0.10


def test_real_adapter_applies_doppler_weighting():
    """Receding audio must build belief far slower than approaching audio.

    Checked in the first few ticks: both eventually pin to the 0.7 audio-only
    cap, so the x0.3 weighting only shows before saturation.
    """
    approaching = RealFusionAdapter(CFG, ["N", "E", "S", "W"])
    receding    = RealFusionAdapter(CFG, ["N", "E", "S", "W"])
    for k in range(4):
        t = k * 0.1
        approaching.feed_audio(AudioDet("N", 0.0, 0.9, receding=False), t)
        approaching.tick(t)
        receding.feed_audio(AudioDet("N", 0.0, 0.9, receding=True), t)
        receding.tick(t)
    # x0.3 weighting -> receding belief is a small fraction of approaching
    assert receding.belief("N") < approaching.belief("N") * 0.6


def test_real_adapter_fires_on_genuine_dual_modal():
    """Sanity: real audio + real vision DOES fire (the gate isn't just stuck off)."""
    eng = RealFusionAdapter(CFG, ["N", "E", "S", "W"])
    fired_any = False
    for k in range(60):
        t = k * 0.1
        eng.feed_audio(AudioDet("N", 0.0, 0.95, receding=False), t)
        eng.feed_vision(VisionDet("N", 0.92), t)
        if eng.tick(t) == "N":
            fired_any = True
    assert fired_any


# ----------------------------------------------------------------------
# The harness has teeth: a leak is actually caught
# ----------------------------------------------------------------------

def test_evaluate_flags_a_benign_leak():
    leaky = ScenarioResult(name="phone_speaker", expected=0, guard="x", seeds=3,
                           fires_per_seed=[0, 1, 0])
    ok, msgs = evaluate([leaky], CFG)
    assert not ok and "phone_speaker" in msgs[0]


def test_evaluate_flags_unbounded_spoof():
    runaway = ScenarioResult(name="spoof_flood", expected=-1, guard="rate_limit",
                             seeds=2, fires_per_seed=[99, 80])
    ok, msgs = evaluate([runaway], CFG)
    assert not ok and "exceeds rate cap" in msgs[0]


def test_reference_model_also_passes():
    """The toy baseline should agree -- 0 benign fires, spoof bounded."""
    ref = run_all(CFG, seeds=6, use_real=False)
    ok, _ = evaluate(ref, CFG)
    assert ok


# ----------------------------------------------------------------------
# Prior-art comparison: a naive immediate-preemption baseline
# ----------------------------------------------------------------------

def test_naive_adapter_fires_on_audio_alone():
    """The strawman fires on a loud siren with no camera -- exactly what the
    cross-modal gate is supposed to stop."""
    naive = NaiveFusionAdapter(CFG, ["N", "E", "S", "W"])
    fired = []
    for k in range(10):
        t = k * 0.1
        naive.feed_audio(AudioDet("N", 0.0, 0.9, receding=False), t)
        if naive.tick(t):
            fired.append(t)
    assert fired                                  # it false-fires
    # ...and the REAL engine does not, on the identical input
    real = RealFusionAdapter(CFG, ["N", "E", "S", "W"])
    real_fired = []
    for k in range(10):
        t = k * 0.1
        real.feed_audio(AudioDet("N", 0.0, 0.9, receding=False), t)
        if real.tick(t):
            real_fired.append(t)
    assert not real_fired


def test_naive_fires_once_per_episode_not_per_tick():
    """Rising-edge latch: continuous detection = one episode, not 100 fires."""
    naive = NaiveFusionAdapter(CFG, ["N", "E", "S", "W"])
    fires = 0
    for k in range(50):
        naive.feed_audio(AudioDet("N", 0.0, 0.95, receding=False), k * 0.1)
        if naive.tick(k * 0.1):
            fires += 1
    assert fires == 1


def test_comparison_shows_gates_eliminate_false_fires():
    comp = run_comparison(CFG, seeds=8)
    # naive leaks in every benign scenario; ours in none
    assert comp["naive_scenarios_leaking"] == comp["benign_scenarios"]
    assert comp["naive_benign_false_fires"] > 0
    assert comp["gated_benign_false_fires"] == 0
