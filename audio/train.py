#!/usr/bin/env python3
"""
Train CRNN for siren detection (ROOT-safe paths)

Data sources (mixable):
  synthetic  audio/data/processed/{train,val}/{positive,negative}/*.wav
             (from audio/prepare_data.py -- SNR-mixed synthetic sirens)
  real       audio/data/real/manifest.csv
             (from audio/prepare_real_data.py -- EVSS, sireNNet, LSSiren,
              UrbanSound8K hard negatives; source-level train/val split)

Model selection: best ROC-AUC on the REAL validation split (falls back to
synthetic val when no real data is present).  Precision/recall are reported
on the val set at 0.5 and at the best-F1 threshold; the best-F1 threshold
is stored in the checkpoint and used downstream by audio/infer.py.

Usage:
    python audio/train.py                          # synthetic + real, 50 epochs
    python audio/train.py --no-synthetic           # real only
    python audio/train.py --no-real                # legacy synthetic-only
    python audio/train.py --epochs 5 --max-samples 400   # quick smoke run
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- project-local imports ---
from model import SirenCRNN, count_parameters
from dataset import AudioDataset, ManifestAudioDataset, collate_fn
from preprocess import AudioPreprocessor

# Anchor to project root: .../greenwave
ROOT = Path(__file__).resolve().parents[1]
CFG_DEFAULT = ROOT / "common" / "config.yaml"
PROC_DIR = ROOT / "audio" / "data" / "processed"
REAL_MANIFEST_DEFAULT = ROOT / "audio" / "data" / "real" / "manifest.csv"
LOG_DIR = ROOT / "logs" / "audio"
CKPT_DIR = ROOT / "checkpoints"


class Trainer:
    """Training manager for siren detection"""

    def __init__(self, config_path: str = str(CFG_DEFAULT)):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create directories (ROOT-safe)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Preprocessor (explicit config path)
        self.preprocessor = AudioPreprocessor(config_path)

        # Model
        self.model = SirenCRNN(
            n_mels=self.config['audio']['n_mels'],
            conv_channels=[32, 64, 128],
            rnn_hidden=128,
            rnn_layers=2,
            dropout=0.3
        ).to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")

        # Loss/opt/scheduler
        # NOTE: SirenCRNN returns sigmoid probs. If you switch model to return logits,
        # change to BCEWithLogitsLoss() and remove sigmoid in model.forward().
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

        # Metrics buffers
        self.train_losses, self.val_losses, self.val_aucs = [], [], []
        self.val_precisions, self.val_recalls = [], []
        self.best_auc = 0.0
        self.last_thresh = 0.5

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_data(
        self,
        use_synthetic: bool = True,
        use_real: bool = True,
        real_manifest: Path = REAL_MANIFEST_DEFAULT,
        batch_size: int = 32,
        num_workers: int = 0,
        max_samples: int | None = None,
    ):
        """Build mixed train loader + a primary val loader.

        Primary val = real split (deployment-representative) when available,
        else synthetic val.  A secondary val loader is kept for reporting.
        """
        train_sets, self.val_primary, self.val_secondary = [], None, None
        self.val_primary_name = ""

        if use_real and Path(real_manifest).exists():
            train_sets.append(ManifestAudioDataset(
                str(real_manifest), self.preprocessor, split="train",
                augment=True, max_samples=max_samples))
            self.val_primary = ManifestAudioDataset(
                str(real_manifest), self.preprocessor, split="val",
                augment=False, max_samples=max_samples)
            self.val_primary_name = "real"
        elif use_real:
            print(f"[WARN] real manifest not found: {real_manifest}"
                  f"\n  Run: python audio/prepare_real_data.py --download --build")

        if use_synthetic and (PROC_DIR / "train").exists():
            train_sets.append(AudioDataset(
                str(PROC_DIR / "train"), self.preprocessor,
                augment=True, max_samples=max_samples))
            synth_val = AudioDataset(
                str(PROC_DIR / "val"), self.preprocessor,
                augment=False, max_samples=max_samples)
            if self.val_primary is None:
                self.val_primary, self.val_primary_name = synth_val, "synthetic"
            else:
                self.val_secondary = synth_val
        elif use_synthetic:
            print(f"[WARN] synthetic data not found: {PROC_DIR / 'train'}"
                  f"\n  Run: python audio/prepare_data.py")

        if not train_sets or self.val_primary is None:
            raise FileNotFoundError("No training data found -- prepare data first")

        self.train_dataset = (ConcatDataset(train_sets)
                              if len(train_sets) > 1 else train_sets[0])
        print(f"Train windows: {len(self.train_dataset)}  "
              f"| primary val ({self.val_primary_name}): {len(self.val_primary)}")

        # persistent_workers keeps Windows-spawned workers alive across epochs
        # (audio decode + mel extraction dominate, so workers matter a lot)
        kw = dict(collate_fn=collate_fn, num_workers=num_workers,
                  pin_memory=True, persistent_workers=num_workers > 0)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **kw)
        self.val_loader = DataLoader(
            self.val_primary, batch_size=batch_size, shuffle=False, **kw)
        self.val_loader_secondary = (
            DataLoader(self.val_secondary, batch_size=batch_size, shuffle=False, **kw)
            if self.val_secondary else None)

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        total = 0.0
        pbar = tqdm(self.train_loader, desc="Training")
        for specs, labels in pbar:
            specs = specs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(specs)             # probs in [0,1], shape (B,1)
            loss = self.criterion(outputs, labels)  # BCE on probabilities
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total / max(1, len(self.train_loader))

    @torch.no_grad()
    def _collect(self, loader):
        """Run the model over a loader; return (avg_loss, preds, labels)."""
        self.model.eval()
        total, preds, labs = 0.0, [], []
        for specs, labels in tqdm(loader, desc="Validation", leave=False):
            specs = specs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(specs)
            total += self.criterion(outputs, labels).item()
            preds.extend(outputs.detach().cpu().numpy().ravel())
            labs.extend(labels.detach().cpu().numpy().ravel())
        avg_loss = total / max(1, len(loader))
        return avg_loss, np.asarray(preds, np.float32), np.asarray(labs, np.float32)

    def validate(self):
        """Validate on the primary val set.

        Returns (loss, auc, precision@thr, recall@thr, best_f1_threshold).
        Precision/recall are computed at the best-F1 threshold.
        """
        avg_loss, preds, labs = self._collect(self.val_loader)

        if len(np.unique(labs)) < 2:
            return avg_loss, 0.0, 0.0, 0.0, 0.5

        auc = roc_auc_score(labs, preds)

        precision, recall, thresholds = precision_recall_curve(labs, preds)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        best = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
        thr = float(thresholds[best]) if len(thresholds) else 0.5
        self.last_thresh = thr

        p_at = precision_score(labs, preds >= thr, zero_division=0)
        r_at = recall_score(labs, preds >= thr, zero_division=0)
        return avg_loss, float(auc), float(p_at), float(r_at), thr

    def train(self, epochs: int = 50, early_stop_patience: int = 10):
        print("=" * 60)
        print("Starting training...")
        print("=" * 60)
        patience = 0
        epoch = 0
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            tr_loss = self.train_epoch()
            va_loss, va_auc, va_p, va_r, thr = self.validate()

            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)
            self.val_aucs.append(va_auc)
            self.val_precisions.append(va_p)
            self.val_recalls.append(va_r)

            print(f"Train Loss: {tr_loss:.4f}")
            print(f"Val({self.val_primary_name}) Loss: {va_loss:.4f} | AUC: {va_auc:.4f}")
            print(f"Precision: {va_p:.4f} | Recall: {va_r:.4f}  (threshold={thr:.4f})")

            self.scheduler.step(va_auc)

            if va_auc > self.best_auc:
                self.best_auc = va_auc
                self.save_checkpoint(CKPT_DIR / "audio_best.pt", epoch, va_auc, thr,
                                     precision=va_p, recall=va_r)
                print(f"[OK] Saved best model (AUC {va_auc:.4f})")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print("Early stopping.")
                    break

        # Final save + report
        self.save_checkpoint(CKPT_DIR / "audio_final.pt", epoch,
                             self.val_aucs[-1], self.last_thresh,
                             precision=self.val_precisions[-1],
                             recall=self.val_recalls[-1])
        self.save_training_history()
        self.plot_training_curves()
        self.final_report()

    def final_report(self):
        """Reload the best checkpoint and report final val metrics."""
        best_path = CKPT_DIR / "audio_best.pt"
        if not best_path.exists():
            return
        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        thr = float(ckpt.get("threshold", 0.5))

        print("\n" + "=" * 60)
        print(f"FINAL REPORT (best checkpoint, epoch {ckpt.get('epoch')})")
        print("=" * 60)
        for name, loader in [(f"val[{self.val_primary_name}]", self.val_loader),
                             ("val[synthetic]", self.val_loader_secondary)]:
            if loader is None:
                continue
            _, preds, labs = self._collect(loader)
            if len(np.unique(labs)) < 2:
                continue
            auc = roc_auc_score(labs, preds)
            for t in (0.5, thr):
                p = precision_score(labs, preds >= t, zero_division=0)
                r = recall_score(labs, preds >= t, zero_division=0)
                print(f"  {name:16s} thr={t:.3f}  AUC={auc:.4f}  "
                      f"precision={p:.4f}  recall={r:.4f}")
        print(f"\n[OK] Best AUC: {self.best_auc:.4f}  "
              f"(checkpoint: {best_path})")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, epoch: int, auc: float, threshold: float,
                        precision: float = 0.0, recall: float = 0.0):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "auc": auc,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "config": self.config
        }, path)

    def save_training_history(self):
        hist = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_aucs": self.val_aucs,
            "val_precisions": self.val_precisions,
            "val_recalls": self.val_recalls,
            "best_auc": self.best_auc
        }
        (LOG_DIR).mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "training_history.json", "w") as f:
            json.dump(hist, f, indent=2)

    def plot_training_curves(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        # Loss
        axes[0].plot(self.train_losses, label="Train")
        axes[0].plot(self.val_losses, label="Val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True); axes[0].legend()
        # AUC
        axes[1].plot(self.val_aucs, label="Val AUC")
        axes[1].axhline(y=self.best_auc, linestyle="--", label=f"Best: {self.best_auc:.4f}")
        axes[1].set_title("AUC"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC"); axes[1].grid(True); axes[1].legend()
        # Precision / recall
        axes[2].plot(self.val_precisions, label="Precision")
        axes[2].plot(self.val_recalls, label="Recall")
        axes[2].set_title("Val precision/recall @best-F1 thr"); axes[2].set_xlabel("Epoch"); axes[2].grid(True); axes[2].legend()
        plt.tight_layout()
        out = LOG_DIR / "training_curves.png"
        plt.savefig(out, dpi=150)
        print(f"[OK] Saved training curves to {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-manifest', default=str(REAL_MANIFEST_DEFAULT))
    parser.add_argument('--no-real',      action='store_true',
                        help='train on synthetic data only')
    parser.add_argument('--no-synthetic', action='store_true',
                        help='train on real data only')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--config',     default=str(CFG_DEFAULT))
    parser.add_argument('--num_workers', type=int, default=4)  # spawn-safe: main() is __main__-guarded
    parser.add_argument('--max-samples', type=int, default=None,
                        help='cap samples per dataset (smoke testing)')
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config)
    trainer.load_data(
        use_synthetic=not args.no_synthetic,
        use_real=not args.no_real,
        real_manifest=Path(args.real_manifest),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    trainer.train(epochs=args.epochs, early_stop_patience=8)


if __name__ == "__main__":
    main()
