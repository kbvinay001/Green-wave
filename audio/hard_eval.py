#!/usr/bin/env python3
"""
Deployment-realistic hard test for the siren CRNN -- Green Wave++ (T5)

WHY
---
The model reports AUC = 1.000 on the held-out validation split. That split is
clean, isolated recordings -- sirens from the siren datasets vs noise from
UrbanSound8K, each on its own. Separating a clean siren clip from a clean
drilling clip is an easy task; 1.000 proves the *task* was easy, not that the
detector survives a street.

This builds the test the deployment actually faces: the SAME held-out siren
recordings, but mixed into UrbanSound8K street noise at low SNR (-5..+5 dB).
Lower SNR is the distance proxy -- a far siren is quiet relative to the
traffic around the mic. The honest number this produces (expect ~0.85-0.92)
is the one to report; the 1.000 is the clean-condition ceiling.

NO LEAKAGE
----------
  - Signal  = the held-out VAL siren windows (never trained on).
  - Noise   = UrbanSound8K clips from ambient street classes, with every
              slice already used anywhere in the train/val manifests removed.
  - The siren 'siren'/'gun_shot' US8K classes are excluded as backgrounds
    (the first is a positive; the second is impulsive, not street ambient).

Run:
    python audio/hard_eval.py                 # full, GPU if available
    python audio/hard_eval.py --max-pos 80    # quick look

Writes evaluation/results/audio_hard_eval.json and
docs/img/audio_hard_eval.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audio"))

from model import SirenCRNN            # noqa: E402
from preprocess import AudioAugmenter, AudioPreprocessor   # noqa: E402

CKPT      = ROOT / "checkpoints" / "audio_best.pt"
MANIFEST  = ROOT / "audio" / "data" / "real" / "manifest_val.csv"
US8K_ROOT = ROOT / "audio" / "data" / "real" / "extracted" / "urbansound8k" / "UrbanSound8K"
SNR_LEVELS = [-5.0, 0.0, 5.0]
# UrbanSound8K ambient classes used as street background (exclude siren + gun_shot)
BG_CLASSES = {"air_conditioner", "car_horn", "children_playing", "dog_bark",
              "drilling", "engine_idling", "jackhammer", "street_music"}
WINDOW_SEC = 3.0
RNG_SEED   = 42


# ---------------------------------------------------------------------------

def load_model(device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = SirenCRNN(n_mels=128, conv_channels=[32, 64, 128],
                      rnn_hidden=128, rnn_layers=2)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, ck


class Scorer:
    """waveform (16k mono) -> p_siren, using the exact training feature path."""

    def __init__(self, model, device):
        self.pre = AudioPreprocessor(config_path=str(ROOT / "common" / "config.yaml"))
        self.model = model
        self.device = device

    def score_many(self, waves: list[np.ndarray], batch: int = 64) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for i in range(0, len(waves), batch):
                chunk = waves[i:i + batch]
                mels = [self.pre.normalize(self.pre.extract_melspec(w)) for w in chunk]
                T = max(m.shape[1] for m in mels)
                padded = [np.pad(m, ((0, 0), (0, T - m.shape[1]))) for m in mels]
                x = torch.from_numpy(np.stack(padded)).unsqueeze(1).float().to(self.device)
                p = self.model(x).squeeze(1).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)


def used_us8k_slices() -> set[str]:
    used = set()
    for mf in ("manifest_train.csv", "manifest_val.csv"):
        p = ROOT / "audio" / "data" / "real" / mf
        if not p.exists():
            continue
        for row in csv.DictReader(open(p)):
            m = re.search(r"urbansound8k_(.+?)_\d+\.wav", row.get("filepath", ""))
            if m:
                used.add(m.group(1) + ".wav")
    return used


def background_pool(pre: AudioPreprocessor, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Fresh US8K street-noise clips (no siren, none seen in training)."""
    used = used_us8k_slices()
    meta = list(csv.DictReader(open(US8K_ROOT / "metadata" / "UrbanSound8K.csv")))
    cands = [r for r in meta
             if r["class"] in BG_CLASSES and r["slice_file_name"] not in used]
    rng.shuffle(cands)
    waves, want = [], n
    for r in cands:
        if len(waves) >= want:
            break
        f = US8K_ROOT / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
        if not f.exists():
            continue
        try:
            waves.append(pre.load_audio(str(f)))
        except Exception:
            continue
    return waves


def fixed_len(aug: AudioAugmenter, w: np.ndarray) -> np.ndarray:
    return aug.random_segment(w, int(WINDOW_SEC * aug.sr))


def main():
    ap = argparse.ArgumentParser(description="Hard (noisy, low-SNR) siren test")
    ap.add_argument("--max-pos", type=int, default=0, help="cap siren windows (0=all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(RNG_SEED)
    model, ck = load_model(device)
    scorer = Scorer(model, device)
    pre = scorer.pre
    aug = AudioAugmenter(sample_rate=pre.sr)
    thr = float(ck.get("threshold", 0.5))
    print(f"[hard-eval] device={device}  stored clean AUC={ck.get('auc')}  threshold={thr:.3f}")

    # --- load held-out val windows -----------------------------------------
    root = MANIFEST.parent
    pos_files, neg_files = [], []
    for row in csv.DictReader(open(MANIFEST)):
        (pos_files if row["label"] == "1" else neg_files).append(root / row["filepath"])
    if args.max_pos:
        pos_files = pos_files[:args.max_pos]
    print(f"[hard-eval] held-out val: {len(pos_files)} siren, {len(neg_files)} noise windows")

    pos_waves = [fixed_len(aug, pre.load_audio(str(p))) for p in pos_files]
    val_neg_waves = [fixed_len(aug, pre.load_audio(str(p))) for p in neg_files]

    # --- (0) reproduce the clean number, to prove the pipeline is faithful --
    clean_scores = scorer.score_many(pos_waves + val_neg_waves)
    clean_labels = np.array([1] * len(pos_waves) + [0] * len(val_neg_waves))
    clean_auc = roc_auc_score(clean_labels, clean_scores)
    print(f"[hard-eval] reproduced CLEAN val AUC = {clean_auc:.4f}  "
          f"(checkpoint stored {ck.get('auc')})")

    # --- build the hard set: held-out sirens mixed into fresh street noise --
    n_bg = max(len(pos_waves), 320)
    bg = background_pool(pre, n_bg, rng)
    bg = [fixed_len(aug, w) for w in bg]
    print(f"[hard-eval] {len(bg)} fresh US8K street-noise clips (none seen in training)")

    neg_scores = scorer.score_many(bg)            # background-only = label 0

    results = {"clean_auc": round(float(clean_auc), 4),
               "stored_clean_auc": ck.get("auc"),
               "threshold": thr, "per_snr": {}, "snr_levels": SNR_LEVELS}
    roc_curves = {}

    all_pos_scores, all_pos_at = [], []
    for snr in SNR_LEVELS:
        mixed = []
        for i, sw in enumerate(pos_waves):
            nw = bg[i % len(bg)]
            m, _ = aug.set_snr(sw, nw, snr)
            mixed.append(m)
        ps = scorer.score_many(mixed)
        all_pos_scores.append(ps)
        all_pos_at.append(snr)

        scores = np.concatenate([ps, neg_scores])
        labels = np.array([1] * len(ps) + [0] * len(neg_scores))
        preds = (scores >= thr).astype(int)
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_curves[snr] = (fpr.tolist(), tpr.tolist())
        results["per_snr"][f"{snr:+g}"] = {
            "auc":       round(float(auc), 4),
            "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
            "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
            "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
            "n_pos": int(len(ps)), "n_neg": int(len(neg_scores)),
        }
        r = results["per_snr"][f"{snr:+g}"]
        print(f"  SNR {snr:+.0f} dB :  AUC {r['auc']:.3f}   "
              f"P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}")

    # overall across all SNRs
    pos_all = np.concatenate(all_pos_scores)
    scores = np.concatenate([pos_all, neg_scores])
    labels = np.array([1] * len(pos_all) + [0] * len(neg_scores))
    preds = (scores >= thr).astype(int)
    results["overall"] = {
        "auc":       round(float(roc_auc_score(labels, scores)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
    }
    o = results["overall"]
    print(f"[hard-eval] OVERALL hard set:  AUC {o['auc']:.3f}  "
          f"P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}")

    out = ROOT / "evaluation" / "results" / "audio_hard_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[hard-eval] wrote {out}")

    _plot(results, roc_curves, clean_auc, ROOT / "docs" / "img" / "audio_hard_eval.png")


def _plot(results, roc_curves, clean_auc, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    snrs = results["snr_levels"]
    aucs = [results["per_snr"][f"{s:+g}"]["auc"] for s in snrs]
    ax1.axhline(clean_auc, color="#888", linestyle="--", label=f"clean ceiling ({clean_auc:.3f})")
    ax1.plot(snrs, aucs, "o-", color="#b0413e", linewidth=1.8, markersize=7, label="siren in street noise")
    for s, a in zip(snrs, aucs):
        ax1.annotate(f"{a:.3f}", (s, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax1.set_xlabel("signal-to-noise ratio (dB)  --  lower = farther siren")
    ax1.set_ylabel("ROC AUC")
    ax1.set_title("Siren detection degrades gracefully with SNR")
    ax1.set_xticks(snrs)
    ax1.set_ylim(0.5, 1.02)
    ax1.legend(loc="lower right")

    for snr in snrs:
        fpr, tpr = roc_curves[snr]
        ax2.plot(fpr, tpr, linewidth=1.5, label=f"{snr:+.0f} dB (AUC {results['per_snr'][f'{snr:+g}']['auc']:.3f})")
    ax2.plot([0, 1], [0, 1], color="#ccc", linestyle=":")
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.set_title("ROC by noise level")
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[hard-eval] wrote {path}")


if __name__ == "__main__":
    main()
