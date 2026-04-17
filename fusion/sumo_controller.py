#!/usr/bin/env python3
"""
Traffic signal preemption controller -- Green Wave++

Wraps TraCI to execute the green-wave sequence:
  1. All-red clearance (flush conflicting phases)
  2. Green cascade -- each signal clears just before the vehicle arrives
  3. Natural phase plan restored after configurable hold duration

Falls back to MockTLSController (no SUMO/TraCI dependency) when TraCI
is not importable, so the rest of the pipeline works on any machine.
Callers don't need to know which backend is active.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Dict, List, Optional


class TLSPhase(Enum):
    RED    = "red"
    YELLOW = "yellow"
    GREEN  = "green"


# ---------------------------------------------------------------------------
# Mock backend (no SUMO required)
# ---------------------------------------------------------------------------

class MockTLSController:
    """
    In-memory traffic light state machine.
    Used when SUMO/TraCI is not installed or mock=True is passed.
    """

    def __init__(self, tls_ids: List[str]):
        self._states: Dict[str, TLSPhase] = {t: TLSPhase.GREEN for t in tls_ids}
        self._lock = threading.Lock()

    def set_phase(self, tls_id: str, phase: TLSPhase) -> None:
        with self._lock:
            self._states[tls_id] = phase

    def get_phase(self, tls_id: str) -> TLSPhase:
        with self._lock:
            return self._states.get(tls_id, TLSPhase.GREEN)

    def get_all_states(self) -> Dict[str, str]:
        with self._lock:
            return {tid: ph.value for tid, ph in self._states.items()}


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class SumoController:
    """
    Green-wave controller.  Transparently uses TraCI when available, otherwise
    falls back to the mock controller.

    The preemption sequence runs in a daemon thread so the fusion loop
    (caller) is never blocked by the all-red sleep or green hold.

    Simultaneous preemptions on different approaches are allowed; each
    runs its own sequence thread.
    """

    def __init__(self, config: dict, sumo_cfg: Optional[str] = None, mock: bool = False):
        sc = config["sumo"]
        self._step_s       = float(sc["step_length"])
        self._all_red_s    = float(sc["all_red_duration"])
        self._green_hold_s = float(sc["preempt_green_duration"])

        # Try TraCI unless caller forces mock
        self._traci = None
        self._mock  = True
        if not mock:
            try:
                import traci
                self._traci = traci
                self._mock  = False
            except ImportError:
                pass

        all_tls = self._collect_tls_ids(config)
        self._mock_ctrl = MockTLSController(all_tls)

        self._active: Dict[str, float] = {}   # lane_id -> activation timestamp
        self._lock = threading.Lock()

        print(f"[OK] SumoController ready  ({'mock' if self._mock else 'TraCI'} backend)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, sumo_cfg: Optional[str] = None) -> None:
        if self._mock or self._traci is None or not sumo_cfg:
            return
        self._traci.start(["sumo", "-c", sumo_cfg, "--step-length", str(self._step_s)])

    def stop(self) -> None:
        if not self._mock and self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                pass

    def step(self) -> None:
        """Advance the simulation by one step (SUMO only; no-op in mock)."""
        if not self._mock and self._traci is not None:
            self._traci.simulationStep()

    # ------------------------------------------------------------------
    # Preemption API
    # ------------------------------------------------------------------

    def trigger_preemption(
        self,
        lane_id:      str,
        corridor_tls: List[str],
        eta_seconds:  List[float],
    ) -> None:
        """
        Fire the green-wave sequence for a corridor.
        Safe to call from any thread; sequence runs in a daemon thread.
        Duplicate calls for the same lane while active are ignored.
        """
        with self._lock:
            if lane_id in self._active:
                return
            self._active[lane_id] = time.time()

        t = threading.Thread(
            target=self._sequence,
            args=(lane_id, corridor_tls, eta_seconds),
            daemon=True,
            name=f"preempt-{lane_id}",
        )
        t.start()

    def release(self, lane_id: str) -> None:
        """Manually release a preemption (e.g. vehicle cancelled or passed early)."""
        with self._lock:
            self._active.pop(lane_id, None)

    def is_active(self, lane_id: str) -> bool:
        with self._lock:
            return lane_id in self._active

    def get_tls_states(self) -> Dict[str, str]:
        if self._mock:
            return self._mock_ctrl.get_all_states()

        states: Dict[str, str] = {}
        if self._traci is not None:
            try:
                for tls_id in self._traci.trafficlight.getIDList():
                    # Phase indices vary by network; even=green is the common convention
                    idx = self._traci.trafficlight.getPhase(tls_id)
                    states[tls_id] = "green" if idx % 2 == 0 else "red"
            except Exception:
                pass
        return states

    # ------------------------------------------------------------------
    # Sequence (runs in background thread)
    # ------------------------------------------------------------------

    def _sequence(self, lane_id: str, corridor_tls: List[str], eta_seconds: List[float]) -> None:
        """
        Full preemption sequence:
          1. All TLS in corridor -> red (clearance)
          2. Per-signal green at each ETA (minus clearance time already elapsed)
          3. Hold configured duration, then restore red and release lock
        """
        print(f"[!!] Preemption: {lane_id} -> {corridor_tls}")

        # All-red clearance
        for tls_id in corridor_tls:
            self._set(tls_id, TLSPhase.RED)
        time.sleep(self._all_red_s)
        elapsed = self._all_red_s

        # Green cascade -- each signal clears at its ETA
        for idx, tls_id in enumerate(corridor_tls):
            target_eta = eta_seconds[idx] if idx < len(eta_seconds) else elapsed
            remaining  = max(0.0, target_eta - elapsed)
            if remaining > 0:
                time.sleep(remaining)
                elapsed += remaining
            self._set(tls_id, TLSPhase.GREEN)
            print(f"   [GREEN] {tls_id} -> green  (t+{elapsed:.1f}s)")

        # Hold green
        time.sleep(self._green_hold_s)

        # Restore
        for tls_id in corridor_tls:
            self._set(tls_id, TLSPhase.RED)

        self.release(lane_id)
        print(f"[OK] Preemption complete: {lane_id}")

    # ------------------------------------------------------------------

    def _set(self, tls_id: str, phase: TLSPhase) -> None:
        if self._mock:
            self._mock_ctrl.set_phase(tls_id, phase)
            return
        if self._traci is None:
            return
        try:
            idx = {TLSPhase.GREEN: 0, TLSPhase.YELLOW: 1, TLSPhase.RED: 2}[phase]
            self._traci.trafficlight.setPhase(tls_id, idx)
        except Exception as e:
            print(f"  [WARN] TraCI set {tls_id}: {e}")

    def _collect_tls_ids(self, config: dict) -> List[str]:
        ids: List[str] = []
        for corridor in config.get("intersection", {}).get("corridors", []):
            for t in corridor.get("intersections", []):
                ids.append(t["id"])
        if not ids:
            # Matches RoutePredictor defaults
            ids = ["J_N1", "J_N2", "J_N3", "J_S1", "J_S2", "J_S3",
                   "J_E1", "J_E2", "J_W1", "J_W2"]
        return ids


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "sumo": {
            "step_length":          0.1,
            "all_red_duration":     2.5,
            "preempt_green_duration": 12.0,
            "downstream_lookahead": 3,
        }
    }

    ctrl = SumoController(cfg, mock=True)
    print("\nInitial TLS states:")
    for tls, state in ctrl.get_tls_states().items():
        print(f"  {tls}: {state}")

    print("\nTriggering preemption on approach_north...")
    ctrl.trigger_preemption(
        lane_id      = "approach_north",
        corridor_tls = ["J_N1", "J_N2", "J_N3"],
        eta_seconds  = [3.0, 5.5, 8.0],
    )

    for i in range(8):
        time.sleep(1)
        states = ctrl.get_tls_states()
        north_states = {k: v for k, v in states.items() if k.startswith("J_N")}
        print(f"  t={i+1}s | {north_states}")

    print("\n[OK] SumoController self-test complete")
