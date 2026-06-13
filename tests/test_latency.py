"""Latency benchmark helpers -- the pure timing math."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.latency import _percentile, time_calls  # noqa: E402


def test_percentile():
    xs = list(range(1, 101))            # 1..100
    assert _percentile(xs, 0) == 1
    assert _percentile(xs, 100) == 100
    assert _percentile(xs, 50) in (50, 51)
    assert _percentile(xs, 95) in (95, 96)


def test_time_calls_structure_and_order():
    out = time_calls(lambda: sum(range(100)), iters=20, warmup=2)
    assert set(out) == {"mean_ms", "median_ms", "p95_ms", "iters"}
    assert out["iters"] == 20
    assert out["p95_ms"] >= out["median_ms"] >= 0


def test_time_calls_measures_a_known_delay():
    # a 5 ms sleep must register as clearly non-trivial (loose bound, CI-safe)
    out = time_calls(lambda: time.sleep(0.005), iters=5, warmup=1)
    assert out["median_ms"] >= 3.0
