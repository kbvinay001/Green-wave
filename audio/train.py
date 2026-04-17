#!/usr/bin/env python3
"""
Train CRNN for siren detection (ROOT-safe paths)
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# --- project-local imports ---
from model import SirenCRNN, count_parameters
from dataset import AudioDataset, collate_fn
from preprocess import AudioPreprocessor

# Anchor to project root: .../greenwave
ROOT = Path(__file__).resolve().parents[1]
CFG_DEFAULT = ROOT / "common" / "config.yaml"
PROC_DIR = ROOT / "audio" / "data" / "processed"
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
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        # Metrics buffers
        self.train_losses, self.val_losses, self.val_aucs = [], [], []
        self.best_auc = 0.0
        self.last_thresh = 0.5

    def load_data(self, train_dir: Path, val_dir: Path, batch_size: int = 32, num_workers: int = 0):
        """Load train/val datasets"""
        self.train_dataset = AudioDataset(str(train_dir), self.preprocessor, augment=True)
        self.val_dataset   = AudioDataset(str(val_dir),   self.preprocessor, augment=False)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )

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

    def validate(self):
        self.model.eval()
        total = 0.0
        preds, labs = [], []
        with torch.no_grad():
            for specs, labels in tqdm(self.val_loader, desc="Validation"):
                specs = specs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                total += loss.item()
                preds.extend(outputs.detach().cpu().numpy().ravel())
                labs.extend(labels.detach().cpu().numpy().ravel())
        avg_loss = total / max(1, len(self.val_loader))

        preds = np.asarray(preds, dtype=np.float32)
        labs  = np.asarray(labs,  dtype=np.float32)
        auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0

        # Precision/recall curve to approximate recall at ~5% FPR (using precision ~ 1 - FPR proxy)
        precision, recall, thresholds = precision_recall_curve(labs, preds)
        fpr_proxy = 1 - precision
        target = 0.05
        idx = int(np.argmin(np.abs(fpr_proxy - target)))
        recall_at_5fpr = float(recall[idx]) if idx < len(recall) else 0.0
        thresh_at_5fpr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        self.last_thresh = thresh_at_5fpr

        return avg_loss, float(auc), recall_at_5fpr, thresh_at_5fpr

    def train(self, epochs: int = 50, early_stop_patience: int = 10):
        print("=" * 60)
        print("Starting training...")
        print("=" * 60)
        patience = 0
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            tr_loss = self.train_epoch()
            va_loss, va_auc, rec5, thr = self.validate()

            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)
            self.val_aucs.append(va_auc)

            print(f"Train Loss: {tr_loss:.4f}")
            print(f"Val   Loss: {va_loss:.4f} | AUC: {va_auc:.4f}")
            print(f"Recall@5%FPR: {rec5:.4f} (threshold={thr:.4f})")

            self.scheduler.step(va_auc)

            if va_auc > self.best_auc:
                self.best_auc = va_auc
                self.save_checkpoint(CKPT_DIR / "audio_best.pt", epoch, va_auc, thr)
                print(f"[OK] Saved best model (AUC {va_auc:.4f})")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print("Early stopping.")
                    break

        # Final save
        self.save_checkpoint(CKPT_DIR / "audio_final.pt", epoch, self.val_aucs[-1], self.last_thresh)
        self.save_training_history()
        self.plot_training_curves()
        print(f"\n[OK] Training complete! Best AUC: {self.best_auc:.4f}")

    def save_checkpoint(self, path: Path, epoch: int, auc: float, threshold: float):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "auc": auc,
            "threshold": threshold,
            "config": self.config
        }, path)

    def save_training_history(self):
        hist = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_aucs": self.val_aucs,
            "best_auc": self.best_auc
        }
        (LOG_DIR).mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "training_history.json", "w") as f:
            json.dump(hist, f, indent=2)

    def plot_training_curves(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Loss
        axes[0].plot(self.train_losses, label="Train")
        axes[0].plot(self.val_losses, label="Val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True); axes[0].legend()
        # AUC
        axes[1].plot(self.val_aucs, label="Val AUC")
        axes[1].axhline(y=self.best_auc, linestyle="--", label=f"Best: {self.best_auc:.4f}")
        axes[1].set_title("AUC"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC"); axes[1].grid(True); axes[1].legend()
        plt.tight_layout()
        out = LOG_DIR / "training_curves.png"
        plt.savefig(out, dpi=150)
        print(f"[OK] Saved training curves to {out}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=str(PROC_DIR / 'train'))
    parser.add_argument('--val_dir',   default=str(PROC_DIR / 'val'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs',     type=int, default=20)
    parser.add_argument('--config',     default=str(CFG_DEFAULT))
    parser.add_argument('--num_workers', type=int, default=0)  # 0 is safest on Windows
    args = parser.parse_args()

    if not Path(args.train_dir).exists():
        print(f"[FAIL] Training data not found: {args.train_dir}")
        print("  Run: python audio\\prepare_data.py")
        return

    trainer = Trainer(config_path=args.config)
    trainer.load_data(Path(args.train_dir), Path(args.val_dir), batch_size=args.batch_size, num_workers=args.num_workers)
    trainer.train(epochs=args.epochs, early_stop_patience=8)

if __name__ == "__main__":
    main()
