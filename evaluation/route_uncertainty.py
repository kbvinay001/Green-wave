#!/usr/bin/env python3
"""
Route-uncertainty test -- Green Wave++ (T8)

The route predictor assumes the ambulance follows the mapped corridor. What
if it doesn't -- turns off after the first junction? The green wave has
already been launched for the WHOLE corridor, so the downstream junctions get
held green for a vehicle that never arrives. Does the system wedge, or
recover?

This runs the real closed-loop pipeline twice per seed on Benz Circle:

  control     the EV completes the corridor (the predictor was right)
  deviation   the EV is rerouted OFF the corridor right after the first
              signal (the predictor was wrong)

and measures the cost of being wrong: civilian delay, and -- the safety
point -- whether every preempted signal RESTORES to normal afterward, so a
bad prediction self-heals instead of stranding the network on green.

The graceful-degradation claim it checks: because each preemption holds for a
fixed `preempt_green_duration` (12 s) then restores, a wrong route costs at
most a bounded burst of misapplied green, never a stuck intersection.

Run:
    python -m evaluation.route_uncertainty                # 8 seeds
    python -m evaluation.route_uncertainty --seeds 1-4 --gui
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.counterfactual import (  # noqa: E402
    EV_ID, EV_ROUTE, EV_TYPE, RUN_DIR, _load_config, _parse_seeds,
    civilian_summary, parse_tripinfo, resolve_command_etas, write_run_cfg,
)
from evaluation.counterfactual import SyntheticSensors      # noqa: E402
from fusion.fuser import TemporalFusionEngine               # noqa: E402
from fusion.route_predictor import RoutePredictor           # noqa: E402
from fusion.sumo_controller import SumoController           # noqa: E402
from integration.pipeline import EndToEndPipeline           # noqa: E402

DEVIATION_EDGE = "415701992"   # branches off just after the first corridor signal


def run(cfg_path: Path, trip_path: Path, *, seed: int, deviate: bool,
        depart: float, duration: float, gui: bool, config: dict) -> dict:
    import traci

    corridor   = config["intersection"]["corridors"][0]
    lane_id    = corridor["lane_id"]
    first_edge = corridor["intersections"][0]["approach_edge"]
    tls_ids    = [t["id"] for t in corridor["intersections"]]

    controller = SumoController(config, sumo_cfg=str(cfg_path), mock=False, gui=gui)
    controller.start()
    predictor = RoutePredictor(config=config)
    lanes   = EndToEndPipeline._lanes_from_config(config)
    fusion  = TemporalFusionEngine(lanes, config)
    sensors = SyntheticSensors(seed, lane_id, float(corridor.get("heading_deg", 270.0)))

    first_edge_len = traci.lane.getLength(f"{first_edge}_0")
    ev_added = triggered = rerouted = False
    trigger_t: Optional[float] = None
    deviated_at: Optional[float] = None
    # per-TLS: did it ever go preempted, and was it released by the end?
    preempted = set()

    t = 0.0
    while t < duration:
        controller.step()
        t = traci.simulation.getTime()

        if not ev_added and t >= depart:
            traci.vehicle.add(EV_ID, EV_ROUTE, typeID=EV_TYPE)
            ev_added = True

        live = ev_added and EV_ID in traci.vehicle.getIDList()
        if live:
            speed = traci.vehicle.getSpeed(EV_ID)
            d = traci.vehicle.getDrivingDistance(EV_ID, first_edge,
                                                 max(first_edge_len - 1.0, 0.0))
            dist = d if d >= 0 else float("inf")
            audio_conf, bearing = sensors.audio(dist)
            commands = fusion.update(audio_conf, bearing, sensors.vision(dist, speed), t)
            for cmd in commands:
                ids, etas = resolve_command_etas(predictor, cmd)
                controller.trigger_preemption(cmd.target_lane, ids, etas)
                if not triggered:
                    triggered, trigger_t = True, round(t, 1)

            # once preemption is up and the EV has cleared the first signal,
            # send it off the corridor (deviation run only)
            if deviate and triggered and not rerouted and trigger_t and t > trigger_t + 5:
                try:
                    traci.vehicle.changeTarget(EV_ID, DEVIATION_EDGE)
                    rerouted, deviated_at = True, round(t, 1)
                except traci.TraCIException:
                    pass

        # signals are "preempted" whenever the controller holds the lane active
        if controller.is_active(lane_id):
            preempted.update(tls_ids)

    released = not controller.is_active(lane_id)   # restored by end of window?
    controller.stop()

    ev, civ = parse_tripinfo(trip_path)
    return {
        "seed": seed, "deviated": deviate,
        "ev_finished": ev is not None,
        "ev_travel_s": round(ev["duration"], 1) if ev else None,
        "triggered_at": trigger_t, "deviated_at": deviated_at,
        "preempted_tls": len(preempted),
        "released_by_end": bool(released),
        "civilians": civilian_summary(civ),
    }


def main():
    ap = argparse.ArgumentParser(description="Route-uncertainty: EV leaves the predicted corridor")
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("--depart", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=700.0)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    cfg = _load_config()
    seeds = _parse_seeds(args.seeds)
    print(f"[route-unc] control vs deviation, seeds={seeds}")

    rows = []
    for seed in seeds:
        rec = {"seed": seed}
        for deviate in (False, True):
            tag = "dev" if deviate else "ctl"
            cfgp, trip = write_run_cfg(RUN_DIR, f"ru_s{seed}_{tag}", seed, scale=1.0)
            r = run(cfgp, trip, seed=seed, deviate=deviate, depart=args.depart,
                    duration=args.duration, gui=args.gui, config=cfg)
            rec[tag] = r
            print(f"  seed {seed} {tag}: EV {'left corridor' if deviate and r['deviated_at'] else 'completed'}"
                  f"  civ_loss={r['civilians']['mean_time_loss_s']}s"
                  f"  released={r['released_by_end']}")
        rows.append(rec)

    # aggregate: cost of a wrong prediction + the safety invariant
    extra = [rec["dev"]["civilians"]["mean_time_loss_s"] - rec["ctl"]["civilians"]["mean_time_loss_s"]
             for rec in rows]
    all_released = all(rec["dev"]["released_by_end"] and rec["ctl"]["released_by_end"] for rec in rows)
    deviated_ok = sum(1 for rec in rows if rec["dev"]["deviated_at"] is not None)
    summary = {
        "seeds": seeds,
        "deviation_took_effect_in": f"{deviated_ok}/{len(rows)}",
        "mean_extra_civilian_loss_s": round(statistics.mean(extra), 2),
        "max_extra_civilian_loss_s": round(max(extra), 2),
        "all_preemptions_released": all_released,
    }
    out = ROOT / "evaluation" / "results" / "route_uncertainty.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n[route-unc] ===== summary =====")
    print(f"  deviation took effect in {summary['deviation_took_effect_in']} seeds")
    print(f"  every preemption released by end: {summary['all_preemptions_released']}  "
          f"(the safety invariant -- no stuck greens)")
    print(f"  cost of a wrong route: +{summary['mean_extra_civilian_loss_s']}s mean civilian "
          f"delay (max +{summary['max_extra_civilian_loss_s']}s)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
