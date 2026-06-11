#!/usr/bin/env python3
"""
Traffic signal preemption controller -- Green Wave++

Executes the green-wave sequence:
  1. All-red clearance (flush conflicting phases)
  2. Green cascade -- each signal clears just before the vehicle arrives
  3. Natural phase plan restored after configurable hold duration

Two backends, same public API:

  MOCK   (default)  In-memory TLS state machine driven by wall-clock daemon
                    threads.  Used for demos and any machine without SUMO.

  TraCI             Real SUMO control.  TraCI simulation time only advances
                    on simulationStep(), so the cascade is NOT slept through
                    wall-clock: trigger_preemption() builds a schedule in
                    SIMULATION time and step() applies whatever is due.

                    Signals are commanded with setRedYellowGreenState using
                    each junction's controlled-link list: links whose inbound
                    edge is the EV approach get 'G', everything else 'r'.
                    The original signal program is restored afterwards via
                    setProgram, so normal operation resumes automatically.

Approach edges come from config:
    intersection:
      corridors:
        - lane_id: approach_north
          intersections:
            - {id: J_N1, distance_m: 0,   approach_edge: edgeIntoJ_N1}
            - {id: J_N2, distance_m: 100, approach_edge: edgeIntoJ_N2}

A TLS without a known approach_edge falls back to all-red + restore (safe
clearance, no green wave) and logs a warning once.
"""

from __future__ import annotations

import heapq
import os
import threading
import time
from enum import Enum
from pathlib import Path
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
    Green-wave controller.  Mock backend unless a SUMO config is supplied
    (and TraCI imports), in which case real signals are commanded.

    Mock mode: sequences run in daemon threads (wall-clock).
    TraCI mode: sequences are scheduled in sim time; call step() each tick.
    """

    def __init__(self, config: dict, sumo_cfg: Optional[str] = None,
                 mock: bool = False, gui: bool = False):
        sc = config["sumo"]
        self._step_s       = float(sc["step_length"])
        self._all_red_s    = float(sc["all_red_duration"])
        self._green_hold_s = float(sc["preempt_green_duration"])
        self._gui          = gui
        self._sumo_cfg     = sumo_cfg or sc.get("cfg") or None

        # tls_id -> inbound approach edge (per corridor config)
        self._approach_edge: Dict[str, str] = {}
        for corridor in config.get("intersection", {}).get("corridors", []):
            for t in corridor.get("intersections", []):
                if t.get("approach_edge"):
                    self._approach_edge[t["id"]] = t["approach_edge"]

        # Try TraCI unless caller forces mock
        self._traci = None
        self._mock  = True
        if not mock and self._sumo_cfg:
            try:
                import traci
                self._traci = traci
                self._mock  = False
            except ImportError:
                print("[WARN] traci not importable -- falling back to mock")

        all_tls = self._collect_tls_ids(config)
        self._mock_ctrl = MockTLSController(all_tls)

        self._active: Dict[str, float] = {}   # lane_id -> activation timestamp
        self._lock = threading.Lock()

        # TraCI scheduling state
        self._schedule: list = []              # heap of (sim_time, seq, fn)
        self._seq = 0
        self._saved_programs: Dict[str, str] = {}
        self._warned_no_edge: set = set()

        print(f"[OK] SumoController ready  ({'mock' if self._mock else 'TraCI'} backend"
              f"{', cfg=' + str(self._sumo_cfg) if not self._mock else ''})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, sumo_cfg: Optional[str] = None) -> None:
        """Launch SUMO and connect TraCI (no-op in mock mode)."""
        if self._mock or self._traci is None:
            return
        cfg = sumo_cfg or self._sumo_cfg
        binary = self._sumo_binary()
        self._traci.start([binary, "-c", str(cfg),
                           "--step-length", str(self._step_s),
                           "--quit-on-end", "true"])
        print(f"[OK] SUMO started: {Path(cfg).name} ({Path(binary).stem})")

    def stop(self) -> None:
        if not self._mock and self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                pass

    def step(self) -> None:
        """Advance one sim step and apply due preemption actions (TraCI)."""
        if self._mock or self._traci is None:
            return
        self._traci.simulationStep()
        now = self._traci.simulation.getTime()

        # Pop everything that's due first, run it after dropping the lock.
        # finish() takes the lock again via release(), and a plain Lock
        # deadlocks if we're still holding it here.
        due = []
        with self._lock:
            while self._schedule and self._schedule[0][0] <= now:
                _, _, fn = heapq.heappop(self._schedule)
                due.append(fn)
        for fn in due:
            try:
                fn()
            except Exception as e:
                print(f"  [WARN] scheduled TLS action failed: {e}")

    def sim_time(self) -> float:
        if self._mock or self._traci is None:
            return time.time()
        return float(self._traci.simulation.getTime())

    def _sumo_binary(self) -> str:
        name = "sumo-gui" if self._gui else "sumo"
        home = os.environ.get("SUMO_HOME", "")
        for cand in (Path(home) / "bin" / f"{name}.exe",
                     Path(home) / "bin" / name):
            if cand.exists():
                return str(cand)
        return name   # hope it's in PATH

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
        Mock: daemon thread with wall-clock sleeps.
        TraCI: actions scheduled in simulation time, applied by step().
        Duplicate calls for the same lane while active are ignored.
        """
        with self._lock:
            if lane_id in self._active:
                return
            self._active[lane_id] = time.time()

        if self._mock or self._traci is None:
            t = threading.Thread(
                target=self._sequence_mock,
                args=(lane_id, corridor_tls, eta_seconds),
                daemon=True,
                name=f"preempt-{lane_id}",
            )
            t.start()
        else:
            self._schedule_traci(lane_id, corridor_tls, eta_seconds)

    def release(self, lane_id: str) -> None:
        """Manually release a preemption (e.g. vehicle cancelled or passed early)."""
        with self._lock:
            self._active.pop(lane_id, None)

    def is_active(self, lane_id: str) -> bool:
        with self._lock:
            return lane_id in self._active

    # ------------------------------------------------------------------
    # State reporting (dashboard)
    # ------------------------------------------------------------------

    def get_tls_states(self) -> Dict[str, str]:
        if self._mock:
            return self._mock_ctrl.get_all_states()

        states: Dict[str, str] = {}
        if self._traci is None:
            return states
        try:
            for tls_id in self._traci.trafficlight.getIDList():
                state = self._traci.trafficlight.getRedYellowGreenState(tls_id)
                edge = self._approach_edge.get(tls_id)
                if edge:
                    states[tls_id] = self._approach_color(tls_id, state, edge)
                else:
                    # majority colour as a coarse dashboard signal
                    g = sum(c in "Gg" for c in state)
                    y = sum(c in "Yy" for c in state)
                    states[tls_id] = ("green" if g >= len(state) / 2
                                      else "yellow" if y > 0 else "red")
        except Exception:
            pass
        return states

    def _approach_color(self, tls_id: str, state: str, edge: str) -> str:
        """Colour of the signal controlling the EV approach edge."""
        try:
            links = self._traci.trafficlight.getControlledLinks(tls_id)
            for idx, group in enumerate(links):
                for (in_lane, _out, _via) in group:
                    if in_lane.rsplit("_", 1)[0] == edge:
                        c = state[idx]
                        return ("green" if c in "Gg"
                                else "yellow" if c in "Yy" else "red")
        except Exception:
            pass
        return "red"

    # ------------------------------------------------------------------
    # Mock sequence (wall-clock, daemon thread)
    # ------------------------------------------------------------------

    def _sequence_mock(self, lane_id: str, corridor_tls: List[str],
                       eta_seconds: List[float]) -> None:
        print(f"[!!] Preemption: {lane_id} -> {corridor_tls}")

        for tls_id in corridor_tls:
            self._mock_ctrl.set_phase(tls_id, TLSPhase.RED)
        time.sleep(self._all_red_s)
        elapsed = self._all_red_s

        # Green cascade -- each signal clears at its ETA
        for idx, tls_id in enumerate(corridor_tls):
            target_eta = eta_seconds[idx] if idx < len(eta_seconds) else elapsed
            remaining  = max(0.0, target_eta - elapsed)
            if remaining > 0:
                time.sleep(remaining)
                elapsed += remaining
            self._mock_ctrl.set_phase(tls_id, TLSPhase.GREEN)
            print(f"   [GREEN] {tls_id} -> green  (t+{elapsed:.1f}s)")

        time.sleep(self._green_hold_s)

        for tls_id in corridor_tls:
            self._mock_ctrl.set_phase(tls_id, TLSPhase.RED)

        self.release(lane_id)
        print(f"[OK] Preemption complete: {lane_id}")

    # ------------------------------------------------------------------
    # TraCI sequence (sim-time scheduled, applied in step())
    # ------------------------------------------------------------------

    def _schedule_traci(self, lane_id: str, corridor_tls: List[str],
                        eta_seconds: List[float]) -> None:
        now = self._traci.simulation.getTime()
        print(f"[!!] Preemption (sim t={now:.1f}s): {lane_id} -> {corridor_tls}")

        # Remember original programs (once per TLS, restored at the end)
        for tls_id in corridor_tls:
            if tls_id not in self._saved_programs:
                try:
                    self._saved_programs[tls_id] = \
                        self._traci.trafficlight.getProgram(tls_id)
                except Exception:
                    self._saved_programs[tls_id] = "0"

        def all_red(tls_id: str):
            def fn():
                n = len(self._traci.trafficlight.getRedYellowGreenState(tls_id))
                self._traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n)
            return fn

        def green_wave(tls_id: str):
            def fn():
                state = self._green_state_for(tls_id)
                if state is None:
                    if tls_id not in self._warned_no_edge:
                        self._warned_no_edge.add(tls_id)
                        print(f"  [WARN] {tls_id}: no approach_edge in config -- "
                              f"holding all-red instead of green wave")
                    return
                self._traci.trafficlight.setRedYellowGreenState(tls_id, state)
                print(f"   [GREEN] {tls_id} -> EV approach green "
                      f"(sim t={self._traci.simulation.getTime():.1f}s)")
            return fn

        def restore(tls_id: str):
            def fn():
                prog = self._saved_programs.get(tls_id, "0")
                try:
                    self._traci.trafficlight.setProgram(tls_id, prog)
                except Exception as e:
                    print(f"  [WARN] restore {tls_id}: {e}")
            return fn

        def finish():
            self.release(lane_id)
            print(f"[OK] Preemption complete: {lane_id} "
                  f"(sim t={self._traci.simulation.getTime():.1f}s)")

        with self._lock:
            # 1. immediate all-red clearance on the whole corridor
            for tls_id in corridor_tls:
                self._push(now, all_red(tls_id))
            # 2. each TLS turns green at its own ETA (never before clearance ends)
            last_green = now
            for idx, tls_id in enumerate(corridor_tls):
                eta = eta_seconds[idx] if idx < len(eta_seconds) else 0.0
                t_green = now + max(eta, self._all_red_s)
                last_green = max(last_green, t_green)
                self._push(t_green, green_wave(tls_id))
            # 3. hold, then restore the normal plans
            t_restore = last_green + self._green_hold_s
            for tls_id in corridor_tls:
                self._push(t_restore, restore(tls_id))
            self._push(t_restore, finish)

    def _push(self, sim_time: float, fn) -> None:
        self._seq += 1
        heapq.heappush(self._schedule, (sim_time, self._seq, fn))

    def _green_state_for(self, tls_id: str) -> Optional[str]:
        """State string: 'G' on links fed by the EV approach edge, 'r' elsewhere."""
        edge = self._approach_edge.get(tls_id)
        if not edge:
            return None
        links = self._traci.trafficlight.getControlledLinks(tls_id)
        chars = []
        hit = False
        for group in links:
            green = any(in_lane.rsplit("_", 1)[0] == edge
                        for (in_lane, _out, _via) in group)
            hit = hit or green
            chars.append("G" if green else "r")
        return "".join(chars) if hit else None

    # ------------------------------------------------------------------

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
# Self-test (mock backend)
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
