#!/usr/bin/env python3
"""
End-to-end latency benchmark -- Green Wave++ (T7)

Reviewers and deployers ask one blunt question: does this run in real time on
commodity hardware, and how long from siren onset to the light changing? This
measures it on the ACTUAL trained models and the real per-tick code paths, in
wall-clock milliseconds -- not sim-time.

Two numbers, kept separate because they answer different questions:

  COMPUTE latency  -- how long each real component takes per call, vs the
                      real-time budget its thread has. This is the
                      "can the hardware keep up" number.
  ACTION delay     -- siren onset -> preemption command, dominated by the
                      DELIBERATE certainty gating (0.5 s arm-hold + the
                      belief accumulation + cross-modal confirmation), not by
                      compute. This is a design latency, reported honestly so
                      nobody confuses it with slowness.

Per-component real-time budgets (from config.yaml):
  audio  -- one chunk per hop_sec (0.1 s) -> 100 ms budget
  vision -- fps_target 25 -> 40 ms budget
  fusion -- 10 Hz loop -> 100 ms budget

Run:
    python evaluation/latency.py                 # full, GPU if available
    python evaluation/latency.py --iters 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))

CFG = ROOT / "common" / "config.yaml"
CKPT = ROOT / "checkpoints" / "audio_best.pt"
YOLO_W = ROOT / "vision" / "weights" / "yolov11s-ambulance.pt"
SR = 16000


def _percentile(xs, q):
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))
    return s[k]


def time_calls(fn, iters, warmup=5):
    """Return per-call ms: mean / median / p95 over `iters` (after warmup)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms":   round(statistics.mean(samples), 4),
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms":    round(_percentile(samples, 95), 4),
        "iters":     iters,
    }


def _siren_chunk():
    """One 1 s, 3-channel siren-ish chunk so the full audio path (CRNN + GCC-
    PHAT + Doppler) actually runs -- the worst case, not the silent fast path."""
    import csv
    import librosa
    manifest = ROOT / "audio" / "data" / "real" / "manifest_val.csv"
    wav = None
    if manifest.exists():
        for row in csv.DictReader(open(manifest)):
            if row["label"] == "1":
                wav = manifest.parent / row["filepath"]
                break
    if wav and wav.exists():
        y, _ = librosa.load(str(wav), sr=SR, mono=True)
    else:                                   # fallback: a 700 Hz wail
        t = np.arange(SR) / SR
        y = (0.3 * np.sin(2 * np.pi * (700 + 100 * np.sin(2 * np.pi * 2 * t)) * t)).astype(np.float32)
    y = y[:SR] if len(y) >= SR else np.pad(y, (0, SR - len(y)))
    # three mics: small integer-sample delays give a non-zero TDOA
    return [y, np.roll(y, 3), np.roll(y, 6)]


def main():
    ap = argparse.ArgumentParser(description="Green Wave++ latency benchmark")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[latency] device={device} ({gpu})  iters={args.iters}")

    results = {"device": device, "gpu": gpu, "iters": args.iters,
               "budgets_ms": {"audio": 100.0, "vision": 40.0, "fusion": 100.0},
               "components": {}}

    # --- audio: full real path (CRNN siren + GCC-PHAT bearing + Doppler) ----
    from stream_detector import StreamingDetector
    det = StreamingDetector(model_path=str(CKPT), config_path=str(CFG))
    chunk = _siren_chunk()

    def audio_call():
        det.reset() if hasattr(det, "reset") else None
        det.process_multichannel(chunk)
    results["components"]["audio_full"] = time_calls(audio_call, args.iters)
    print(f"  audio  (CRNN+GCC-PHAT+Doppler) : {results['components']['audio_full']['median_ms']} ms median")

    # --- vision: YOLOv11s one frame ----------------------------------------
    try:
        from ultralytics import YOLO
        ymodel = YOLO(str(YOLO_W))
        frame = (np.random.default_rng(0).integers(0, 255, (640, 640, 3))).astype(np.uint8)

        def vision_call():
            ymodel.predict(frame, imgsz=640, device=0 if device == "cuda" else "cpu", verbose=False)
        results["components"]["vision_yolo"] = time_calls(vision_call, args.iters, warmup=8)
        print(f"  vision (YOLOv11s 640)         : {results['components']['vision_yolo']['median_ms']} ms median")
    except Exception as e:  # noqa: BLE001
        results["components"]["vision_yolo"] = {"error": str(e)}
        print(f"  vision: skipped ({e})")

    # --- fusion: one 10 Hz tick --------------------------------------------
    import yaml
    from fusion.fuser import Lane, TemporalFusionEngine
    cfg = yaml.safe_load(CFG.read_text())
    lanes = [Lane("approach_north", 0.0, ["A", "B", "C"]),
             Lane("approach_south", 180.0, ["D", "E"]),
             Lane("approach_east", 90.0, ["F", "G"]),
             Lane("approach_west", 270.0, ["H", "I"])]
    eng = TemporalFusionEngine(lanes, cfg)
    vdets = [{"lane_id": "approach_north", "confidence": 0.9, "approaching": True,
              "distance_m": 80.0, "speed_kmh": 50.0, "speed_mps": 13.9}]
    _tick = [0.0]

    def fusion_call():
        _tick[0] += 0.1
        eng.update(0.8, 5.0, vdets, _tick[0])
    results["components"]["fusion_tick"] = time_calls(fusion_call, max(args.iters, 200))
    print(f"  fusion (one 10 Hz tick)       : {results['components']['fusion_tick']['median_ms']} ms median")

    # --- verdicts: real-time factor per thread -----------------------------
    def factor(comp, budget):
        c = results["components"].get(comp, {})
        # median can read 0.0 for sub-microsecond ops; fall back to mean
        m = c.get("median_ms") or c.get("mean_ms")
        if not m:
            return None
        return round(budget / m, 1)

    results["realtime_factor"] = {
        "audio":  factor("audio_full", 100.0),
        "vision": factor("vision_yolo", 40.0),
        "fusion": factor("fusion_tick", 100.0),
    }

    # --- action delay (design latency, not compute) ------------------------
    fc = cfg["fusion"]
    results["action_delay"] = {
        "arm_hold_sec": fc.get("arm_duration_sec", 0.5),
        "note": ("siren onset -> preemption is dominated by the deliberate "
                 "certainty gating: the lane must arm (belief >= arm_threshold), "
                 "hold for arm_duration_sec, cross preempt_threshold AND have a "
                 "camera confirmation within vision_confirm_window_sec. In the "
                 "synthetic demo this lands ~5-6 s after onset; the compute "
                 "above is a negligible fraction of it."),
    }

    out = ROOT / "evaluation" / "results" / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n[latency] real-time factor (budget / median compute):")
    for k, v in results["realtime_factor"].items():
        print(f"    {k:<7}: {v}x" + ("  (>1 = keeps up)" if v else "  (n/a)"))
    print(f"[latency] wrote {out}")

    _plot(results, ROOT / "docs" / "img" / "latency.png")


def _plot(results, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    comps = [("audio_full", "audio\n(CRNN+bearing)", 100.0),
             ("vision_yolo", "vision\n(YOLOv11s)", 40.0),
             ("fusion_tick", "fusion\n(10 Hz tick)", 100.0)]
    names, meds, budgets = [], [], []
    for key, label, budget in comps:
        c = results["components"].get(key, {})
        m = c.get("median_ms") or c.get("mean_ms")
        if m is not None:
            # floor tiny values so the bar/label render (sub-resolution ops)
            names.append(label); meds.append(max(m, 0.05)); budgets.append(budget)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = range(len(names))
    ax.bar(x, budgets, width=0.6, color="#e8e8e8", label="real-time budget", zorder=1)
    ax.bar(x, meds, width=0.6, color="#3e8e41", label="measured (median)", zorder=2)
    for xi, m, b in zip(x, meds, budgets):
        ax.text(xi, m, f"{m:.2g} ms\n({b/m:.0f}x)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x), names)
    ax.set_ylabel("latency per call (ms)")
    ax.set_title(f"Compute latency vs real-time budget  ({results['gpu']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[latency] wrote {path}")


if __name__ == "__main__":
    main()
