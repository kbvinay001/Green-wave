#!/usr/bin/env python3
"""
Temporal fusion engine -- Green Wave++

Maintains per-lane belief scores driven by two independent evidence streams:
  - Audio: bearing-weighted siren confidence (Gaussian angle kernel)
  - Vision: direct lane assignments from YOLOv11 detections

Beliefs decay exponentially so stale evidence doesn't linger.  Once a
lane's belief clears arm_threshold and holds for arm_duration_sec, and
then crosses preempt_threshold, a FusionCommand is emitted.

Two ambulances on orthogonal approaches are handled independently; each
fires its own command.  It is the SUMO controller's job to sequence them.

Lane lifecycle:
    IDLE -> ARMED (belief ≥ arm_threshold, held arm_duration_sec)
         -> ACTIVE (preempt_threshold crossed -> FusionCommand emitted)
         -> COOLING (belief decayed after vehicle passed)
         -> IDLE
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class LanePhase(Enum):
    IDLE    = auto()
    ARMED   = auto()   # threshold held long enough, waiting for preempt level
    ACTIVE  = auto()   # preemption command sent, green wave running
    COOLING = auto()   # vehicle passed, belief draining toward reset


@dataclass
class Lane:
    name:            str
    heading_deg:     float        # compass bearing of approach (0=N, 90=E, ...)
    corridor_tls:    List[str]    # TLS IDs in preemption order, nearest first


@dataclass
class FusionCommand:
    """Emitted once when a lane crosses the preemption threshold."""
    target_lane:  str
    corridor_tls: List[str]
    eta_seconds:  List[float]   # rough internal estimate; the pipeline
                                # recomputes against RoutePredictor's real
                                # corridor distances when speed/distance exist
    belief:       float
    timestamp:    float
    speed_mps:    Optional[float] = None   # from the last vision detection
    distance_m:   Optional[float] = None   # distance to the first junction


@dataclass
class LaneState:
    phase:            LanePhase = LanePhase.IDLE
    belief:           float = 0.0
    last_update:      float = field(default_factory=time.time)
    arm_start:        Optional[float] = None   # when belief first reached arm_threshold
    active_since:     Optional[float] = None
    last_speed_mps:   Optional[float] = None   # cached from last vision detection
    last_distance_m:  Optional[float] = None
    last_vision_time: Optional[float] = None   # when the camera last confirmed this lane


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class TemporalFusionEngine:
    """
    Multi-modal, multi-lane belief fusion.

    Parameters come from config.yaml under the 'fusion' key.  The engine
    is intentionally config-driven so deployment parameters can be tuned
    without touching code.
    """

    def __init__(self, lanes: List[Lane], config: dict):
        self.lanes  = {l.name: l for l in lanes}
        self.states = {l.name: LaneState() for l in lanes}

        fc = config["fusion"]
        self.decay_factor      = float(fc["decay_factor"])
        self.sigma_angle       = float(fc["sigma_angle_deg"])
        self.arm_threshold     = float(fc["arm_threshold"])
        self.arm_duration      = float(fc["arm_duration_sec"])
        self.preempt_threshold = float(fc["preempt_threshold"])
        self.preempt_threshold_approaching = float(
            fc.get("preempt_threshold_approaching", self.preempt_threshold * 0.875)
        )
        self.min_visual_speed  = float(fc.get("min_visual_speed_kmh", 30.0))

        # Cross-modal gate: without a recent camera confirmation the belief
        # is capped below the preemption threshold, so audio alone can arm
        # a lane (and earn the graded green extension) but never fire the
        # full green wave.
        self.audio_only_cap    = float(fc.get("audio_only_belief_cap", 0.7))
        self.vision_window     = float(fc.get("vision_confirm_window_sec", 3.0))

        # Lanes that newly ARMED since the caller last asked. The pipeline
        # drains this to fire the graded green extension.
        self.pending_arm_events: List[str] = []

        # Audio fusion gain -- tuned so a single bearing hit at σ=0 takes
        # ~4 ticks to cross arm_threshold at full confidence.
        self._audio_gain  = 0.22
        self._vision_gain = 0.35

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        audio_conf:        float,
        audio_bearing:     Optional[float],
        vision_detections: List[dict],
        timestamp:         float,
    ) -> List[FusionCommand]:
        """
        Single fusion tick.  Returns any newly triggered preemption commands.

        vision_detections: list of dicts with keys:
            lane_id, confidence, approaching, distance_m, speed_kmh, speed_mps
        """
        self._decay(timestamp)

        if audio_conf > 0 and audio_bearing is not None:
            self._fuse_audio(audio_conf, audio_bearing)

        for det in vision_detections:
            self._fuse_vision(det, timestamp)

        self._clamp_beliefs(timestamp)
        return self._evaluate_triggers(timestamp)

    def get_beliefs(self) -> Dict[str, float]:
        return {k: round(v.belief, 4) for k, v in self.states.items()}

    def get_phases(self) -> Dict[str, str]:
        return {k: v.phase.name for k, v in self.states.items()}

    def reset_lane(self, lane_name: str) -> None:
        """Force a lane back to IDLE -- called after green wave completes."""
        if lane_name in self.states:
            self.states[lane_name] = LaneState(last_update=time.time())

    def reset_all(self) -> None:
        now = time.time()
        for name in self.states:
            self.states[name] = LaneState(last_update=now)

    # ------------------------------------------------------------------
    # Internal machinery
    # ------------------------------------------------------------------

    def _decay(self, now: float) -> None:
        for state in self.states.values():
            dt = max(0.01, now - state.last_update)
            state.belief *= self.decay_factor ** dt
            state.last_update = now

    def _bearing_weight(self, audio_bearing: float, lane_heading: float) -> float:
        """Gaussian kernel on angular difference.  Returns [0, 1]."""
        diff = abs(audio_bearing - lane_heading) % 360
        if diff > 180:
            diff = 360 - diff
        return math.exp(-0.5 * (diff / self.sigma_angle) ** 2)

    def _fuse_audio(self, conf: float, bearing: float) -> None:
        for name, lane in self.lanes.items():
            w = self._bearing_weight(bearing, lane.heading_deg)
            self.states[name].belief += conf * w * self._audio_gain

    def _fuse_vision(self, det: dict, timestamp: float) -> None:
        lane_id = det.get("lane_id")
        if lane_id not in self.states:
            return

        state     = self.states[lane_id]
        det_conf  = float(det.get("confidence", 0.5))
        approaching = bool(det.get("approaching", False))
        speed_kmh = det.get("speed_kmh") or det.get("speed_mps", 0) * 3.6
        distance_m = det.get("distance_m")

        if not approaching:
            return
        if speed_kmh < self.min_visual_speed:
            return

        # this counts as a camera confirmation for the cross-modal gate
        state.last_vision_time = timestamp

        # Bigger bump when belief is still building; taper off once near threshold
        gain = self._vision_gain if state.belief < self.preempt_threshold_approaching else 0.12
        state.belief += det_conf * gain

        if det.get("speed_mps"):
            state.last_speed_mps = float(det["speed_mps"])
        elif speed_kmh:
            state.last_speed_mps = speed_kmh / 3.6

        if distance_m is not None:
            state.last_distance_m = float(distance_m)

    def _clamp_beliefs(self, now: float) -> None:
        for state in self.states.values():
            cap = 1.0
            # no camera confirmation recently -> audio-only cap applies
            if (state.last_vision_time is None
                    or now - state.last_vision_time > self.vision_window):
                cap = self.audio_only_cap
            state.belief = max(0.0, min(cap, state.belief))

    def vision_confirmed(self, lane_id: str, now: float) -> bool:
        """True when the camera has seen this lane within the confirm window."""
        t = self.states[lane_id].last_vision_time
        return t is not None and now - t <= self.vision_window

    def pop_arm_events(self) -> List[str]:
        """Lanes that armed since the last call (each arm cycle fires once)."""
        events, self.pending_arm_events = self.pending_arm_events, []
        return events

    def _evaluate_triggers(self, now: float) -> List[FusionCommand]:
        commands = []

        for name, state in self.states.items():

            if state.phase == LanePhase.ACTIVE:
                # Disarm once the siren is gone and belief has drained
                if state.belief < self.arm_threshold * 0.4:
                    state.phase    = LanePhase.COOLING
                    state.arm_start = None
                continue

            if state.phase == LanePhase.COOLING:
                if state.belief < 0.08:
                    state.phase = LanePhase.IDLE
                continue

            # IDLE or ARMED
            if state.belief >= self.arm_threshold:
                if state.arm_start is None:
                    state.arm_start = now
                    state.phase = LanePhase.ARMED
                    self.pending_arm_events.append(name)

                held = now - state.arm_start
                if held >= self.arm_duration and state.belief >= self.preempt_threshold:
                    cmd = self._build_command(name, state, now)
                    state.phase       = LanePhase.ACTIVE
                    state.active_since = now
                    commands.append(cmd)
            else:
                if state.phase == LanePhase.ARMED:
                    # Dropped back below threshold -- reset arm timer
                    state.phase     = LanePhase.IDLE
                    state.arm_start = None

        return commands

    def _build_command(self, lane_name: str, state: LaneState, now: float) -> FusionCommand:
        lane = self.lanes[lane_name]

        # ETA to each corridor TLS
        if state.last_speed_mps and state.last_distance_m:
            eta_base   = state.last_distance_m / max(state.last_speed_mps, 1.0)
            gap_per_tls = 30.0 / max(state.last_speed_mps, 5.0)
        else:
            eta_base    = 8.0
            gap_per_tls = 3.0

        etas = [round(eta_base + i * gap_per_tls, 1) for i in range(len(lane.corridor_tls))]

        return FusionCommand(
            target_lane  = lane_name,
            corridor_tls = lane.corridor_tls,
            eta_seconds  = etas,
            belief       = round(state.belief, 4),
            timestamp    = now,
            speed_mps    = state.last_speed_mps,
            distance_m   = state.last_distance_m,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    import time

    cfg = {
        "fusion": {
            "decay_factor":                    0.92,
            "sigma_angle_deg":                 20.0,
            "arm_threshold":                   0.6,
            "arm_duration_sec":                0.5,
            "preempt_threshold":               0.8,
            "preempt_threshold_approaching":   0.7,
            "min_visual_speed_kmh":            30.0,
        }
    }

    lanes = [
        Lane("north", 0.0,   ["TLS_N1", "TLS_N2", "TLS_N3"]),
        Lane("east",  90.0,  ["TLS_E1", "TLS_E2"]),
        Lane("south", 180.0, ["TLS_S1"]),
        Lane("west",  270.0, ["TLS_W1"]),
    ]

    engine = TemporalFusionEngine(lanes, cfg)

    print("=" * 60)
    print("Fusion engine self-test: vehicle approaching from north")
    print("=" * 60)

    t = 0.0
    triggered = False
    for tick in range(40):
        vision = []
        if tick > 8:
            vision = [{
                "lane_id": "north",
                "confidence": 0.82,
                "approaching": True,
                "distance_m": max(20, 200 - tick * 5),
                "speed_kmh": 55.0,
                "speed_mps": 55.0 / 3.6,
            }]

        commands = engine.update(
            audio_conf=0.80,
            audio_bearing=3.0,
            vision_detections=vision,
            timestamp=t,
        )

        beliefs = engine.get_beliefs()
        phases  = engine.get_phases()
        print(
            f"  t={t:4.1f}s | north={beliefs['north']:.3f} [{phases['north'][:4]}]"
            f" | east={beliefs['east']:.3f}"
            + (f" | ** PREEMPT: {commands[0].target_lane}" if commands else "")
        )

        if commands:
            triggered = True
            print(f"     ETAs: {commands[0].eta_seconds}")

        t += 0.15
        time.sleep(0.02)

    print()
    print(f"{'PASS' if triggered else 'FAIL'} Preemption {'triggered' if triggered else 'NOT triggered (check thresholds)'}")
    print("\nDecay test: no input for 3 seconds")
    engine2 = TemporalFusionEngine(lanes, cfg)
    engine2.update(0.9, 2.0, [], 0.0)
    b0 = engine2.get_beliefs()["north"]
    print(f"  t=0.0  north={b0:.3f}")
    engine2.update(0.0, None, [], 3.0)
    b3 = engine2.get_beliefs()["north"]
    print(f"  t=3.0  north={b3:.3f}")
    # 3 seconds at decay 0.92/s should leave ~78% of the belief
    expected = b0 * 0.92 ** 3
    ok = abs(b3 - expected) < 0.02
    print(f"{'PASS' if ok else 'FAIL'}: decayed to {b3:.3f} (expected ~{expected:.3f})")


if __name__ == "__main__":
    _self_test()
