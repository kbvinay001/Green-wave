#!/usr/bin/env python3
"""
Statistical significance of the counterfactual results -- Green Wave++ (T1)

The headline "29-35% faster" hides its spread, and "± std" is not the same
as "significant." This turns the paired SUMO runs into the numbers a panel
actually asks for:

  - paired t-test (baseline vs system, same seed = same traffic), per density
  - 95% confidence interval on the mean time saved (t-based)
  - Cohen's dz -- paired effect size, so "big" isn't just eyeballed
  - a 10k-sample bootstrap CI as a distribution-free cross-check
  - the same treatment for the civilian-delay cost (the honesty metric)

Reads evaluation/results/counterfactual.json (or any --in file with the same
shape) and writes a stats table, significance.json, and a forest plot.

Usage:
    python -m evaluation.significance
    python -m evaluation.significance --in evaluation/results/counterfactual_perfect80.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 12345     # fixed so the CI is reproducible run to run


# ---------------------------------------------------------------------------
# Core statistics (pure -- unit-tested)
# ---------------------------------------------------------------------------

def paired_stats(a: List[float], b: List[float], alpha: float = 0.05) -> dict:
    """
    Paired comparison of two seed-aligned samples. Reports on the difference
    d = a - b (so for travel time, a=baseline b=system gives time *saved* as a
    positive number). Returns mean, t-based CI, paired t-test, and Cohen's dz.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("need two equal-length samples of size >= 2")
    d   = a - b
    n   = d.size
    mean = float(d.mean())
    sd   = float(d.std(ddof=1))
    se   = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci = (mean - tcrit * se, mean + tcrit * se)

    if sd == 0.0:                       # identical every seed -> degenerate
        t_stat, p_val = (math.inf if mean else 0.0), (0.0 if mean else 1.0)
    else:
        t_stat, p_val = stats.ttest_rel(a, b)

    return {
        "n":          n,
        "mean_diff":  round(mean, 2),
        "sd_diff":    round(sd, 2),
        "ci95":       [round(ci[0], 2), round(ci[1], 2)],
        "t":          round(float(t_stat), 3),
        "df":         n - 1,
        "p_value":    float(f"{float(p_val):.2e}") if p_val < 1e-3 else round(float(p_val), 4),
        "cohen_dz":   round(mean / sd, 3) if sd else float("inf"),
        "significant": bool(p_val < alpha),
    }


def bootstrap_ci(a: List[float], b: List[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = BOOTSTRAP_SEED) -> List[float]:
    """Distribution-free percentile CI on the mean of (a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 2), round(float(hi), 2)]


def effect_label(dz: float) -> str:
    """Cohen's conventional bands, for the reader who wants a word."""
    a = abs(dz)
    if a == float("inf") or a >= 1.2: return "very large"
    if a >= 0.8: return "large"
    if a >= 0.5: return "medium"
    if a >= 0.2: return "small"
    return "negligible"


# ---------------------------------------------------------------------------
# Pulling paired series out of a counterfactual.json
# ---------------------------------------------------------------------------

def series(data: dict, scale: float, mode: str,
           *path: str, baseline: str = "baseline") -> Tuple[List[float], List[float]]:
    """
    Seed-aligned (baseline_value, mode_value) for one density and metric.
    Only cells where BOTH ran are kept (so the pairing stays honest).
    """
    base, sysv = [], []
    for cell in data["per_run"]:
        if cell.get("scale") != scale or baseline not in cell or mode not in cell:
            continue
        b, m = cell[baseline], cell[mode]
        for k in path:
            b, m = b[k], m[k]
        base.append(float(b))
        sysv.append(float(m))
    return base, sysv


def analyze(data: dict, modes: Optional[List[str]] = None) -> dict:
    scales = data["params"]["scales"]
    modes = modes or [m for m in data["params"]["modes"] if m != "baseline"]
    out: dict = {"scales": scales, "modes": modes, "by_scale": {}}

    for scale in scales:
        per_mode = {}
        for mode in modes:
            b_t, s_t = series(data, scale, mode, "ev_travel_s")
            if len(b_t) < 2:
                continue
            travel = paired_stats(b_t, s_t)            # baseline - system = saved
            travel["bootstrap_ci95"] = bootstrap_ci(b_t, s_t)
            travel["pct_saved"] = round(100 * travel["mean_diff"]
                                        / (sum(b_t) / len(b_t)), 1)
            travel["effect"] = effect_label(travel["cohen_dz"])

            b_c, s_c = series(data, scale, mode, "civilians", "mean_time_loss_s")
            # system - baseline = extra delay imposed on civilians
            civ = paired_stats(s_c, b_c)
            civ["bootstrap_ci95"] = bootstrap_ci(s_c, b_c)

            per_mode[mode] = {"n": travel["n"],
                              "ev_time_saved_s": travel,
                              "civilian_extra_loss_s": civ}
        out["by_scale"][f"{scale:g}x"] = per_mode
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(analysis: dict) -> None:
    print("\n" + "=" * 92)
    print("PAIRED SIGNIFICANCE  (baseline vs system, same seed = same traffic)")
    print("=" * 92)
    hdr = (f"{'density':<9}{'mode':<12}{'n':<4}{'saved (s)':<12}"
           f"{'95% CI':<20}{'p':<11}{'dz':<8}{'effect'}")
    for label, per_mode in analysis["by_scale"].items():
        print("-" * 92)
        for mode, m in per_mode.items():
            t = m["ev_time_saved_s"]
            ci = f"[{t['ci95'][0]:.0f}, {t['ci95'][1]:.0f}]"
            p  = t["p_value"]
            pstr = f"{p:.1e}" if isinstance(p, float) and p < 1e-3 else f"{p}"
            sig  = "***" if t["significant"] else " ns"
            print(f"{label:<9}{mode:<12}{t['n']:<4}"
                  f"{t['mean_diff']:>6.1f} ({t['pct_saved']:>4.1f}%){'':<1}"
                  f"{ci:<20}{pstr+sig:<11}{t['cohen_dz']:<8.2f}{t['effect']}")
            c = m["civilian_extra_loss_s"]
            print(f"{'':<9}{'  civilians':<12}{'':<4}"
                  f"{c['mean_diff']:>+6.1f} extra   "
                  f"[{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]    "
                  f"{'sig' if c['significant'] else 'n.s.'}")
    print("=" * 92)
    print("dz = Cohen's paired effect size;  *** p<0.05, ns = not significant\n")


def make_forest_plot(analysis: dict, img_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, per_mode in analysis["by_scale"].items():
        for mode, m in per_mode.items():
            t = m["ev_time_saved_s"]
            rows.append((f"{label}  {mode}", t["mean_diff"], t["ci95"]))
    rows.reverse()

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.5))
    ys = range(len(rows))
    colors = ["#3e8e41" if "closedloop" in r[0] else "#d68910" for r in rows]
    for y, (lbl, mean, ci), col in zip(ys, rows, colors):
        ax.plot(ci, [y, y], color=col, linewidth=2.2, solid_capstyle="round")
        ax.plot([mean], [y], "o", color=col, markersize=7)
    ax.axvline(0, color="#888", linewidth=1, linestyle="--")
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.set_xlabel("ambulance corridor time saved (s)  --  95% CI, paired")
    ax.set_title("Green Wave++ : time saved with 95% confidence intervals")
    ax.margins(y=0.08)
    fig.tight_layout()
    out = img_dir / "significance_ci.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Significance of counterfactual results")
    ap.add_argument("--in", dest="infile",
                    default="evaluation/results/counterfactual.json")
    ap.add_argument("--out", default="evaluation/results/significance.json")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    data = json.loads((ROOT / args.infile).read_text()
                      if not Path(args.infile).is_absolute()
                      else Path(args.infile).read_text())
    analysis = analyze(data)
    print_report(analysis)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2))
    print(f"[out] wrote {out}")

    if not args.no_plot:
        p = make_forest_plot(analysis, ROOT / "docs" / "img")
        print(f"[out] wrote {p}")


if __name__ == "__main__":
    main()
