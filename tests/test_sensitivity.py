"""Sensitivity sweep: the testable bits (the SUMO runs are integration-only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.sensitivity import HIT_RATES, _plot  # noqa: E402


def test_hit_rates_are_valid_probabilities():
    assert HIT_RATES == sorted(HIT_RATES)
    assert all(0.0 < r <= 1.0 for r in HIT_RATES)
    assert min(HIT_RATES) <= 0.3            # the sweep actually stresses the low end


def test_plot_renders_from_rows(tmp_path):
    rows = [
        {"audio_hit_rate": 0.2, "mean_saved_s": 63.9, "ci95": [61.5, 66.4], "trigger_rate": 1.0},
        {"audio_hit_rate": 0.6, "mean_saved_s": 63.3, "ci95": [60.4, 66.2], "trigger_rate": 1.0},
        {"audio_hit_rate": 0.95, "mean_saved_s": 63.5, "ci95": [60.2, 66.7], "trigger_rate": 1.0},
    ]
    out = tmp_path / "s.png"
    _plot(rows, 1.0, out)
    assert out.exists() and out.stat().st_size > 0
