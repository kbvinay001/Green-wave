#!/usr/bin/env python3
"""
Event logger -- Green Wave++

Two outputs per session:
  frames_<ts>.csv        Frame-level telemetry (timestamp, audio_conf, vision_count).
                         Written incrementally so it survives crashes.
  session_<ts>.json      Session summary + all preemption events, written on save().

Designed to be called from the 10 Hz fusion loop with negligible overhead.
CSV is flushed every 50 frames (~5s); JSON is written on explicit save() / shutdown.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fusion.fuser import FusionCommand


@dataclass
class PreemptEvent:
    timestamp:    float
    lane:         str
    belief:       float
    corridor_tls: List[str]
    eta_seconds:  List[float]


class E2ELogger:
    """
    Dual-format session logger.

    Usage:
        logger = E2ELogger(output_dir="outputs/e2e_logs")
        logger.log_frame(ts, audio_conf, vision_count)
        logger.log_preempt(fusion_command)
        logger.save()   # call on shutdown
    """

    def __init__(self, output_dir: str = "outputs/e2e_logs"):
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)

        self._session_ts = time.time()
        self._preempts:  List[PreemptEvent] = []
        self._n_frames   = 0
        self._sum_audio  = 0.0
        self._sum_vision = 0
        self._saved      = False

        csv_path = self._out / f"frames_{int(self._session_ts)}.csv"
        self._csv_f   = open(csv_path, "w", newline="", buffering=1)
        self._csv_w   = csv.writer(self._csv_f)
        self._csv_w.writerow(["timestamp", "audio_conf", "vision_count"])

    # ------------------------------------------------------------------

    def log_frame(self, timestamp: float, audio_conf: float, vision_count: int) -> None:
        self._csv_w.writerow([f"{timestamp:.4f}", f"{audio_conf:.4f}", vision_count])
        self._n_frames   += 1
        self._sum_audio  += audio_conf
        self._sum_vision += vision_count
        # Flush every 50 frames to keep OS write load low
        if self._n_frames % 50 == 0:
            self._csv_f.flush()

    def log_preempt(self, cmd: "FusionCommand") -> None:
        evt = PreemptEvent(
            timestamp    = cmd.timestamp,
            lane         = cmd.target_lane,
            belief       = cmd.belief,
            corridor_tls = cmd.corridor_tls,
            eta_seconds  = cmd.eta_seconds,
        )
        self._preempts.append(evt)
        print(f"[LOG] Preemption logged: {evt.lane}  belief={evt.belief:.3f}")

    def save(self) -> None:
        if self._saved:
            return
        self._saved = True
        self._csv_f.flush()
        self._csv_f.close()

        duration = time.time() - self._session_ts
        n        = max(1, self._n_frames)

        report = {
            "session_duration_sec":              round(duration, 2),
            "total_frames":                      self._n_frames,
            "avg_audio_conf":                    round(self._sum_audio / n, 4),
            "avg_vision_detections_per_frame":   round(self._sum_vision / n, 3),
            "preempt_count":                     len(self._preempts),
            "preempt_events":                    [asdict(e) for e in self._preempts],
        }

        out = self._out / f"session_{int(self._session_ts)}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

        print(
            f"[DONE] Session saved -> {out}\n"
            f"   {self._n_frames} frames | {len(self._preempts)} preemptions | {duration:.1f}s"
        )

    def __del__(self):
        # Best-effort flush if save() was never called
        if not self._saved:
            try:
                self._csv_f.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = E2ELogger(output_dir=tmpdir)

        for i in range(100):
            logger.log_frame(time.time(), audio_conf=0.6 + i * 0.003, vision_count=i % 3)

        # Fake a FusionCommand-like object
        class _FakeCmd:
            timestamp    = time.time()
            target_lane  = "approach_north"
            belief       = 0.87
            corridor_tls = ["J_N1", "J_N2"]
            eta_seconds  = [4.0, 7.5]

        logger.log_preempt(_FakeCmd())
        logger.save()

        files = list(Path(tmpdir).iterdir())
        print(f"\nOutput files: {[f.name for f in files]}")
        print("[OK] E2ELogger test complete")
