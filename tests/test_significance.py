"""Significance stats: the pure functions, checked against known values."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.significance import (  # noqa: E402
    analyze, bootstrap_ci, effect_label, paired_stats, series,
)


def test_paired_stats_matches_known_t():
    # a - b = [10, 12, 8, 11, 9]; mean 10, sd 1.5811, n 5
    a = [110, 112, 108, 111, 109]
    b = [100, 100, 100, 100, 100]
    s = paired_stats(a, b)
    assert s["n"] == 5
    assert s["mean_diff"] == 10.0
    assert s["sd_diff"] == pytest.approx(1.58, abs=0.01)
    # t = mean / se = 10 / (1.5811/sqrt(5)) = 14.14
    assert s["t"] == pytest.approx(14.142, abs=0.01)
    assert s["df"] == 4
    assert s["significant"] is True
    assert s["cohen_dz"] == pytest.approx(6.325, abs=0.01)


def test_paired_stats_ci_brackets_mean():
    a = [110, 112, 108, 111, 109]
    b = [100, 100, 100, 100, 100]
    lo, hi = paired_stats(a, b)["ci95"]
    assert lo < 10.0 < hi
    assert lo == pytest.approx(8.04, abs=0.1)
    assert hi == pytest.approx(11.96, abs=0.1)


def test_paired_stats_no_difference_is_not_significant():
    a = [50, 60, 70, 80]
    s = paired_stats(a, a)          # identical -> zero difference
    assert s["mean_diff"] == 0.0
    assert s["significant"] is False
    assert s["p_value"] == 1.0


def test_paired_stats_rejects_mismatched():
    with pytest.raises(ValueError):
        paired_stats([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        paired_stats([1], [1])


def test_bootstrap_ci_is_reproducible_and_brackets():
    a = [110, 112, 108, 111, 109, 113, 107]
    b = [100] * 7
    ci1 = bootstrap_ci(a, b)
    ci2 = bootstrap_ci(a, b)
    assert ci1 == ci2                       # fixed seed
    assert ci1[0] < 10.0 < ci1[1] + 1e-9    # brackets the ~10 mean


def test_effect_label_bands():
    assert effect_label(0.1) == "negligible"
    assert effect_label(0.3) == "small"
    assert effect_label(0.6) == "medium"
    assert effect_label(0.9) == "large"
    assert effect_label(2.0) == "very large"
    assert effect_label(float("inf")) == "very large"


def _fake_counterfactual():
    """Two seeds, one scale, baseline vs closedloop -- enough to exercise analyze()."""
    def cell(seed, base_t, cl_t, base_civ, cl_civ):
        mk = lambda t, civ: {"ev_travel_s": t, "ev_stops": 2, "ev_stopped_s": 0.0,
                             "triggered_at": 90.0,
                             "civilians": {"arrived": 100, "mean_time_loss_s": civ,
                                           "max_time_loss_s": civ * 2, "mean_waiting_s": 0.0}}
        return {"seed": seed, "scale": 1.0,
                "baseline": mk(base_t, base_civ), "closedloop": mk(cl_t, cl_civ)}
    return {"params": {"scales": [1.0], "modes": ["baseline", "closedloop"]},
            "per_run": [cell(1, 180.0, 120.0, 40.0, 41.0),
                        cell(2, 176.0, 118.0, 38.0, 39.0)]}


def test_series_pairs_by_seed():
    data = _fake_counterfactual()
    base, sysv = series(data, 1.0, "closedloop", "ev_travel_s")
    assert base == [180.0, 176.0]
    assert sysv == [120.0, 118.0]


def test_analyze_end_to_end():
    a = analyze(_fake_counterfactual())
    cl = a["by_scale"]["1x"]["closedloop"]
    assert cl["n"] == 2
    assert cl["ev_time_saved_s"]["mean_diff"] == pytest.approx(59.0, abs=0.1)
    assert cl["ev_time_saved_s"]["pct_saved"] == pytest.approx(33.1, abs=0.5)
    # civilians: system loses ~1s more than baseline
    assert cl["civilian_extra_loss_s"]["mean_diff"] == pytest.approx(1.0, abs=0.1)
