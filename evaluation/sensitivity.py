#!/usr/bin/env python3
"""
Sensor sensitivity sweep -- Green Wave++ (T4)

The whole closed-loop result rests on two assumed sensor numbers: audio at
an 85% per-tick hit rate, vision at 90%. A fair question is "what if the
detector is worse than that?" This sweeps the per-tick hit rate of BOTH
modalities together from 20% to 95% and measures how much corridor time the
system still saves -- the deliverable line being "degrades gracefully, and
still saves time well below the nominal detection rate."

Sweeping both matters: audio alone barely degrades (the EV is in range for
dozens of ticks, so even a 40% per-tick rate almost always accumulates
enough to arm), so the binding constraint is the VISION confirmation inside
the 80 m cross-modal window. Dropping both is the honest stress test.

Same paired design as the counterfactual: one baseline per seed, then the
closed-loop system at each hit rate, same traffic (same seed) throughout, so
the only thing changing is detector reliability. All numbers from SUMO
tripinfo.

Run:
    python -m evaluation.sensitivity                       # 12 seeds, 1x demand
    python -m evaluation.sensitivity --seeds 1-8 --scale 3.0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.counterfactual import (  # noqa: E402
    RUN_DIR, _load_config, _parse_seeds, civilian_summary, parse_tripinfo,
    simulate, write_run_cfg,
)
from evaluation.significance import paired_stats  # noqa: E402

HIT_RATES = [0.20, 0.30, 0.40, 0.60, 0.80, 0.95]


def _ev_travel(cfg_path, trip_path, **kw):
    res = simulate(cfg_path, **kw)
    ev, civ = parse_tripinfo(trip_path)
    return (None if ev is None else
            {"travel": round(ev["duration"], 1),
             "triggered": res["triggered_at"] is not None,
             "civ_loss": civilian_summary(civ)["mean_time_loss_s"]})


def main():
    ap = argparse.ArgumentParser(description="Audio detection-rate sensitivity sweep")
    ap.add_argument("--seeds", default="1-12")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--depart", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=700.0)
    ap.add_argument("--rates", default=",".join(str(r) for r in HIT_RATES))
    args = ap.parse_args()

    cfg = _load_config()
    seeds = _parse_seeds(args.seeds)
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    total = len(seeds) * (1 + len(rates))
    print(f"[sens] audio hit-rate sweep {rates}  seeds={seeds}  scale={args.scale:g}  "
          f"({total} SUMO runs)")

    # baseline per seed (detector-independent)
    base = {}
    done = 0
    for seed in seeds:
        cfgp, trip = write_run_cfg(RUN_DIR, f"sens_s{seed}_baseline", seed, scale=args.scale)
        r = _ev_travel(cfgp, trip, mode="baseline", seed=seed, depart=args.depart,
                       duration=args.duration, config=cfg)
        done += 1
        if r is None:
            print(f"  [skip] seed {seed} baseline did not finish")
            continue
        base[seed] = r["travel"]
        print(f"  [{done}/{total}] baseline seed {seed}: {r['travel']}s")

    # closed-loop at each hit rate -- keep (baseline, closedloop) seed-aligned
    rows = []
    for p in rates:
        pairs, trig, civ = [], 0, []
        for seed in seeds:
            if seed not in base:
                continue
            cfgp, trip = write_run_cfg(RUN_DIR, f"sens_s{seed}_p{p:g}", seed, scale=args.scale)
            r = _ev_travel(cfgp, trip, mode="closedloop", seed=seed, depart=args.depart,
                           duration=args.duration, config=cfg, p_audio=p, p_vision=p)
            done += 1
            if r is None:
                continue
            pairs.append((base[seed], r["travel"]))
            trig += int(r["triggered"])
            civ.append(r["civ_loss"])
        n = len(pairs)
        base_v = [b for b, _ in pairs]
        trav_v = [t for _, t in pairs]
        saved = [b - t for b, t in pairs]
        st = paired_stats(base_v, trav_v) if n >= 2 else None
        row = {
            "audio_hit_rate": p,
            "n": n,
            "mean_saved_s": round(statistics.mean(saved), 1) if saved else 0.0,
            "ci95": st["ci95"] if st else None,
            "p_value": st["p_value"] if st else None,
            "trigger_rate": round(trig / n, 2) if n else 0.0,
            "civ_extra_s": round(statistics.mean(civ), 1) if civ else None,
        }
        rows.append(row)
        ci = f" CI{row['ci95']}" if row["ci95"] else ""
        print(f"  hit-rate {p:.0%}:  saves {row['mean_saved_s']:.1f}s{ci}  "
              f"(fired in {row['trigger_rate']:.0%} of runs)")

    out = ROOT / "evaluation" / "results" / "sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scale": args.scale, "seeds": seeds, "rows": rows}, indent=2))
    print(f"[sens] wrote {out}")
    _plot(rows, args.scale, ROOT / "docs" / "img" / "sensitivity.png")


def _plot(rows, scale, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [r["audio_hit_rate"] * 100 for r in rows]
    ys = [r["mean_saved_s"] for r in rows]
    errs = [[(r["mean_saved_s"] - r["ci95"][0]) if r["ci95"] else 0 for r in rows],
            [(r["ci95"][1] - r["mean_saved_s"]) if r["ci95"] else 0 for r in rows]]

    fig, ax1 = plt.subplots(figsize=(8.8, 4.6))
    ax1.errorbar(xs, ys, yerr=errs, marker="o", color="#3e8e41", linewidth=1.8,
                 capsize=4, label="time saved")
    ax1.axhline(0, color="#bbb", linewidth=1, linestyle="--")
    ax1.set_xlabel("detector hit-rate, audio & vision (%)  --  nominal ~85-90%")
    ax1.set_ylabel("ambulance corridor time saved (s)", color="#3e8e41")
    ax1.set_ylim(0, max(ys) * 1.25)
    ax1.set_title(f"Time saved is robust to detector hit-rate ({scale:g}x demand)")
    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.0f}s", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(xs, [r["trigger_rate"] * 100 for r in rows], marker="s", color="#5b7fa6",
             linewidth=1.2, linestyle=":", label="fired (% of runs)")
    ax2.set_ylabel("preemption fired (% of runs)", color="#5b7fa6")
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[sens] wrote {path}")


if __name__ == "__main__":
    main()
