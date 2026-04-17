#!/usr/bin/env python3
"""
Route predictor -- Green Wave++

Maps a detected approach lane to an ordered sequence of downstream
traffic signals (the "green wave corridor") and computes per-signal
ETAs given current vehicle speed and distance.

In production you'd feed this from an OSM or SUMO network graph.
Here corridors are defined declaratively -- either injected in code
(for tests) or read from config.yaml under 'intersection.corridors'.
The default topology is a symmetric 4-way intersection that matches
the rest of the stack's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Intersection:
    """One traffic signal controller on a green-wave corridor."""
    tls_id:     str
    distance_m: float   # cumulative distance from corridor entry point


@dataclass
class Corridor:
    """
    Ordered upstream->downstream signal sequence for one approach.

    distance_m values in each Intersection are cumulative from where
    the vehicle enters the corridor (e.g. 100m upstream of stop-line).
    """
    lane_id:       str
    intersections: List[Intersection]

    def tls_ids(self) -> List[str]:
        return [i.tls_id for i in self.intersections]

    def eta_seconds(self, speed_mps: float, distance_to_entry_m: float) -> List[float]:
        """
        Per-TLS ETAs from 'now', assuming constant speed.

        distance_to_entry_m: how far the vehicle currently is from the
        corridor entry point (i.e. from the first intersection).
        """
        speed = max(speed_mps, 1.0)
        etas  = []
        for inter in self.intersections:
            total_dist = distance_to_entry_m + inter.distance_m
            etas.append(round(total_dist / speed, 1))
        return etas


class RoutePredictor:
    """
    Resolves a lane ID to its preemption corridor and computes ETAs.

    Usage:
        predictor = RoutePredictor(config=cfg)
        result = predictor.resolve("approach_north", speed_mps=15.0, distance_m=140.0)
        if result:
            tls_ids, etas = result
    """

    def __init__(
        self,
        corridors: Optional[List[Corridor]] = None,
        config:    Optional[dict] = None,
    ):
        if corridors:
            self._corridors: Dict[str, Corridor] = {c.lane_id: c for c in corridors}
        elif config and "intersection" in config:
            self._corridors = self._from_config(config["intersection"])
        else:
            self._corridors = self._default_4way()

    # ------------------------------------------------------------------

    def resolve(
        self,
        lane_id:    str,
        speed_mps:  float = 10.0,
        distance_m: float = 150.0,
    ) -> Optional[Tuple[List[str], List[float]]]:
        """
        Returns (tls_ids, eta_seconds) for the given lane, or None if unknown.
        """
        corridor = self._corridors.get(lane_id)
        if corridor is None:
            return None
        return corridor.tls_ids(), corridor.eta_seconds(speed_mps, distance_m)

    def all_lanes(self) -> List[str]:
        return list(self._corridors.keys())

    def summary(self) -> str:
        lines = ["RoutePredictor corridors:"]
        for lane_id, corridor in self._corridors.items():
            tls = " -> ".join(corridor.tls_ids())
            lines.append(f"  {lane_id:20s}  {tls}")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _from_config(self, inter_cfg: dict) -> Dict[str, Corridor]:
        corridors = {}
        for c in inter_cfg.get("corridors", []):
            intersections = [
                Intersection(tls_id=t["id"], distance_m=float(t["distance_m"]))
                for t in c["intersections"]
            ]
            obj = Corridor(lane_id=c["lane_id"], intersections=intersections)
            corridors[obj.lane_id] = obj
        return corridors

    def _default_4way(self) -> Dict[str, Corridor]:
        """
        Default symmetric 4-way intersection.
        Stop-line is at 0m; upstream intersections at 100m, 200m.
        """
        topology = {
            "approach_north": [("J_N1", 0), ("J_N2", 100), ("J_N3", 200)],
            "approach_south": [("J_S1", 0), ("J_S2", 100), ("J_S3", 200)],
            "approach_east":  [("J_E1", 0), ("J_E2", 100)],
            "approach_west":  [("J_W1", 0), ("J_W2", 100)],
        }
        result = {}
        for lane_id, tls_list in topology.items():
            intersections = [Intersection(tls_id=tid, distance_m=d) for tid, d in tls_list]
            result[lane_id] = Corridor(lane_id=lane_id, intersections=intersections)
        return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pred = RoutePredictor()
    print(pred.summary())
    print()

    test_cases = [
        ("approach_north", 15.0, 140.0),
        ("approach_east",  12.0, 200.0),
        ("approach_west",   8.0,  80.0),
        ("unknown_lane",   10.0, 100.0),
    ]

    print(f"{'Lane':<22} {'Speed':>8} {'Dist':>8}   TLS sequence + ETAs")
    print("-" * 72)
    for lane_id, speed, dist in test_cases:
        result = pred.resolve(lane_id, speed_mps=speed, distance_m=dist)
        if result:
            tls_ids, etas = result
            pairs = ", ".join(f"{t}@{e}s" for t, e in zip(tls_ids, etas))
            print(f"  {lane_id:<20} {speed:>6.1f}m/s {dist:>6.0f}m   {pairs}")
        else:
            print(f"  {lane_id:<20}  -> unknown lane (no corridor defined)")

    print("\n[OK] RoutePredictor self-test complete")
