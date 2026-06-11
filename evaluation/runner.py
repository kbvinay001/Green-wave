#!/usr/bin/env python3
"""
Evaluation runner -- Green Wave++

Scans outputs/e2e_logs/ for all session_*.json files produced by real runs,
pairs each with its frames_*.csv, then prints and saves the three output
metrics: wait-time reduction %, throughput improvement %, and
detection-to-preempt latency.

Usage:
    python -m evaluation.runner
    python evaluation/runner.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.metrics import MetricsCollector, SessionMetrics


LOGS_DIR = Path("outputs/e2e_logs")


def _find_pairs() -> list[tuple[Path, Path | None]]:
    """
    Returns (session_json, frames_csv | None) pairs for every session
    that contains at least one preemption event (preempt_count >= 1).
    """
    pairs = []
    for session_path in sorted(LOGS_DIR.glob("session_*.json")):
        data = json.loads(session_path.read_text())
        if data.get("preempt_count", 0) < 1:
            continue   # skip sessions with no preemptions

        # Match the epoch-based timestamp in the filename
        m = re.search(r"session_(\d+)\.json", session_path.name)
        frames_path = None
        if m:
            ts_str = m.group(1)
            candidate = LOGS_DIR / f"frames_{ts_str}.csv"
            if candidate.exists():
                frames_path = candidate

        pairs.append((session_path, frames_path))
    return pairs


def main() -> None:
    print("\n[>>] Green Wave++ Evaluation Runner")
    print(f"     Scanning: {LOGS_DIR.resolve()}\n")

    if not LOGS_DIR.exists():
        print("[WARN] No logs directory found. Run the pipeline first with --demo flag.")
        return

    collector = MetricsCollector("evaluation/results")
    pairs     = _find_pairs()

    if not pairs:
        print("[WARN] No session logs with preemption events found.")
        print("       Run:  python run.py --demo --no-ui")
        print("       Then re-run this script.")
        return

    results: list[SessionMetrics] = []

    for session_path, frames_path in pairs:
        print(f"[+] {session_path.name}", end="")
        if frames_path:
            print(f"  +  {frames_path.name}")
        else:
            print("  (no frame log -- latency N/A)")

        m = collector.compute_from_session(
            session_path    = str(session_path),
            frames_csv_path = str(frames_path) if frames_path else None,
        )
        print(m.summary())
        results.append(m)

    # ----------------------------------------------------------------
    # Aggregate summary across all sessions
    # ----------------------------------------------------------------
    n = len(results)
    if n > 1:
        avg_wait   = sum(r.wait_reduction_pct          for r in results) / n
        avg_thru   = sum(r.throughput_improvement_pct  for r in results) / n
        avg_lat    = [r.detection_latency_s for r in results if r.detection_latency_s]
        avg_lat_v  = sum(avg_lat) / len(avg_lat) if avg_lat else None

        print("=" * 60)
        print(f"  AGGREGATE  ({n} sessions)")
        print(f"  [1] Avg wait-time reduction      : {avg_wait:+.1f}%")
        print(f"  [2] Avg throughput improvement   : {avg_thru:+.1f}%")
        if avg_lat_v:
            print(f"  [3] Avg detection latency        : {avg_lat_v:.2f} s")
        print("=" * 60)

    # ----------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------
    out_path = collector.save_results(results)
    print(f"\n[DONE] Results saved -> {out_path}")
    print(f"       {len(results)} session(s) | columns: wait_reduction_pct, "
          "throughput_improvement_pct, detection_latency_s")


if __name__ == "__main__":
    main()