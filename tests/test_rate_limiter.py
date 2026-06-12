"""Rate limiter: 4 per hour means 4 -- and the window actually slides."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.rate_limiter import SlidingWindowRateLimiter  # noqa: E402


def test_cap_is_enforced():
    rl = SlidingWindowRateLimiter(max_events=4, window_sec=3600)
    assert [rl.allow("approach_west", now=t) for t in (0, 10, 20, 30)] == [True] * 4
    assert rl.allow("approach_west", now=40) is False
    assert rl.remaining("approach_west", now=40) == 0


def test_window_slides():
    rl = SlidingWindowRateLimiter(max_events=4, window_sec=3600)
    for t in (0, 100, 200, 300):
        assert rl.allow("lane", now=t)
    assert not rl.allow("lane", now=3599)
    # 3601: the t=0 event has aged out, one slot frees up
    assert rl.allow("lane", now=3601)
    # ...and only one
    assert not rl.allow("lane", now=3602)


def test_lanes_are_independent():
    rl = SlidingWindowRateLimiter(max_events=4, window_sec=3600)
    for t in (0, 1, 2, 3):
        assert rl.allow("approach_west", now=t)
    assert not rl.allow("approach_west", now=4)
    # a different lane has its own budget
    assert rl.allow("approach_east", now=4)


def test_denied_attempts_do_not_consume_budget():
    rl = SlidingWindowRateLimiter(max_events=2, window_sec=100)
    assert rl.allow("lane", now=0)
    assert rl.allow("lane", now=1)
    for t in range(2, 50):          # hammer it -- denials must not extend the lockout
        assert not rl.allow("lane", now=t)
    assert rl.allow("lane", now=101)  # t=0 aged out exactly as scheduled


def test_retry_after():
    rl = SlidingWindowRateLimiter(max_events=2, window_sec=100)
    rl.allow("lane", now=0)
    rl.allow("lane", now=50)
    assert rl.retry_after("lane", now=60) == 40.0   # t=0 frees at t=100
    assert rl.retry_after("lane", now=101) == 0.0
