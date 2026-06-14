#!/usr/bin/env python3
"""
Second-corridor generalization test -- Green Wave++ (T6)

Every number elsewhere comes from one intersection (Benz Circle, Vijayawada).
"It works on my road" isn't "it generalizes." This runs the SAME closed-loop
evaluation on a second real network built from OpenStreetMap -- Shollinganallur
in Chennai, a denser 10-signal urban stretch with tighter junction spacing
than Benz Circle, i.e. a genuinely different topology.

Pipeline used to build it (all reproducible, see sim/nets/shollinganallur/):
  osmGet.py  -> shollinganallur_bbox.osm.xml
  netconvert (--tls.guess) -> shollinganallur.net.xml  (74 signals)
  shortest-path corridor finder -> a drivable 10-signal route
  randomTrips.py -> background.rou.xml  (540 vehicles)

Then the unchanged evaluation.counterfactual.simulate() -- real
TemporalFusionEngine + SumoController + synthetic sensors -- runs baseline vs
closed-loop, same paired design as Benz.

Run:
    python -m evaluation.second_corridor                 # 12 seeds
    python -m evaluation.second_corridor --seeds 1-4 --gui
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluation.counterfactual as cf   # noqa: E402
from evaluation.counterfactual import (  # noqa: E402
    _load_config, _parse_seeds, civilian_summary, parse_tripinfo, simulate,
)
from evaluation.significance import paired_stats  # noqa: E402

NET_DIR = ROOT / "sim" / "nets" / "shollinganallur"
RUN_DIR = ROOT / "outputs" / "second_corridor"


def shol_config() -> dict:
    """Base config, with the intersection corridor swapped to Shollinganallur."""
    cfg = copy.deepcopy(_load_config())
    corr = json.loads((NET_DIR / "_config_corridor.json").read_text())
    cfg["intersection"] = {"corridors": [corr]}
    return cfg


def write_cfg(name: str, seed: int) -> tuple[Path, Path]:
    """A Shollinganallur .sumocfg + bluelight-free eval EV, mirroring the Benz
    eval setup but on this network."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    edges = json.loads((NET_DIR / "_config_corridor.json").read_text())
    ev_edges = (ROOT / "sim" / "nets" / "shollinganallur" / "ev.rou.xml").read_text()
    # pull the corridor edge string out of the committed ev.rou.xml
    import re
    m = re.search(r'route id="ev_shol_corridor" edges="([^"]+)"', ev_edges)
    edge_str = m.group(1)

    eval_routes = RUN_DIR / f"{name}.eval.rou.xml"
    eval_routes.write_text(f"""<routes>
    <vType id="{cf.EV_TYPE}" vClass="emergency" accel="3.0" decel="5.0"
           length="6.5" maxSpeed="22.0" speedFactor="1.3" guiShape="emergency"/>
    <route id="{cf.EV_ROUTE}" edges="{edge_str}"/>
</routes>
""")
    cfg_path  = RUN_DIR / f"{name}.sumocfg"
    trip_path = RUN_DIR / f"{name}.tripinfo.xml"
    cfg_path.write_text(f"""<configuration>
    <input>
        <net-file value="{(NET_DIR / 'shollinganallur.net.xml').resolve()}"/>
        <route-files value="{(NET_DIR / 'background.rou.xml').resolve()},{eval_routes.resolve()}"/>
    </input>
    <time><begin value="0"/><step-length value="0.1"/></time>
    <random_number><seed value="{seed}"/></random_number>
    <output><tripinfo-output value="{trip_path.resolve()}"/></output>
    <processing><time-to-teleport value="-1"/><ignore-route-errors value="true"/></processing>
    <report><no-step-log value="true"/><duration-log.disable value="true"/></report>
</configuration>
""")
    return cfg_path, trip_path


def run_seed(seed: int, *, depart: float, duration: float, gui: bool, cfg: dict):
    out = {"seed": seed}
    for mode in ("baseline", "closedloop"):
        cfg_path, trip = write_cfg(f"shol_s{seed}_{mode}", seed)
        res = simulate(cfg_path, mode=mode, seed=seed, depart=depart,
                       duration=duration, gui=gui, config=cfg)
        ev, civ = parse_tripinfo(trip)
        if ev is None:
            print(f"  [skip] seed {seed} {mode}: EV didn't finish")
            return None
        out[mode] = {"ev_travel_s": round(ev["duration"], 1),
                     "ev_stops": res["ev_stops"],
                     "triggered_at": res["triggered_at"],
                     "civ_loss": civilian_summary(civ)["mean_time_loss_s"]}
    return out


def main():
    ap = argparse.ArgumentParser(description="Second-corridor (Shollinganallur) generalization test")
    ap.add_argument("--seeds", default="1-12")
    ap.add_argument("--depart", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=700.0)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    cfg = shol_config()
    seeds = _parse_seeds(args.seeds)
    n_sig = len(cfg["intersection"]["corridors"][0]["intersections"])
    print(f"[shol] Shollinganallur, Chennai -- {n_sig}-signal corridor  seeds={seeds}")

    rows = []
    for seed in seeds:
        r = run_seed(seed, depart=args.depart, duration=args.duration, gui=args.gui, cfg=cfg)
        if r is None:
            continue
        rows.append(r)
        b, c = r["baseline"], r["closedloop"]
        print(f"  seed {seed}: {b['ev_travel_s']}s -> {c['ev_travel_s']}s "
              f"(saved {b['ev_travel_s']-c['ev_travel_s']:.1f}s)  "
              f"stops {b['ev_stops']}->{c['ev_stops']}  trig@{c['triggered_at']}")

    if len(rows) < 2:
        raise SystemExit("[shol] too few completed runs")

    base = [r["baseline"]["ev_travel_s"] for r in rows]
    clos = [r["closedloop"]["ev_travel_s"] for r in rows]
    st = paired_stats(base, clos)
    civ_b = statistics.mean(r["baseline"]["civ_loss"] for r in rows)
    civ_c = statistics.mean(r["closedloop"]["civ_loss"] for r in rows)
    summary = {
        "corridor": "shollinganallur_chennai", "signals": n_sig, "n": len(rows),
        "ev_travel_baseline_s": round(statistics.mean(base), 1),
        "ev_travel_closedloop_s": round(statistics.mean(clos), 1),
        "ev_time_saved_s": st["mean_diff"], "ci95": st["ci95"],
        "pct_saved": round(100 * st["mean_diff"] / statistics.mean(base), 1),
        "p_value": st["p_value"], "cohen_dz": st["cohen_dz"],
        "civilian_extra_loss_s": round(civ_c - civ_b, 2),
    }
    out = ROOT / "evaluation" / "results" / "second_corridor.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n[shol] ===== Shollinganallur generalization =====")
    print(f"  EV {summary['ev_travel_baseline_s']}s -> {summary['ev_travel_closedloop_s']}s  "
          f"saves {summary['ev_time_saved_s']}s ({summary['pct_saved']}%) "
          f"CI{summary['ci95']}  p={summary['p_value']}  dz={summary['cohen_dz']}")
    print(f"  civilian extra delay: +{summary['civilian_extra_loss_s']}s")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
