#!/usr/bin/env python3
"""
Sliding-window rate limiter -- Green Wave++

The traffic-engineering worry: a spoofed siren recording or a glitching
detector could hold an intersection hostage by firing preemption after
preemption. Capping how often any single lane may preempt bounds the
worst case no matter what the sensors claim (default: 4 per sliding hour,
from security.rate_limit in config.yaml).

Plain deque-of-timestamps per key. No background threads, no clock
assumptions -- callers may pass their own `now` (the fusion loop passes
its tick timestamp, tests pass whatever they like).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Optional


class SlidingWindowRateLimiter:

    def __init__(self, max_events: int, window_sec: float):
        self.max_events = int(max_events)
        self.window_sec = float(window_sec)
        self._events: Dict[str, deque] = defaultdict(deque)

    def _prune(self, key: str, now: float) -> deque:
        q = self._events[key]
        cutoff = now - self.window_sec
        while q and q[0] <= cutoff:
            q.popleft()
        return q

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        """True records the event and lets it through; False blocks it."""
        now = time.time() if now is None else now
        q = self._prune(key, now)
        if len(q) >= self.max_events:
            return False
        q.append(now)
        return True

    def remaining(self, key: str, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        return max(0, self.max_events - len(self._prune(key, now)))

    def retry_after(self, key: str, now: Optional[float] = None) -> float:
        """Seconds until the oldest event ages out and a slot frees. 0 = open now."""
        now = time.time() if now is None else now
        q = self._prune(key, now)
        if len(q) < self.max_events:
            return 0.0
        return q[0] + self.window_sec - now
