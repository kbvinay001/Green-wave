#!/usr/bin/env python3
"""
PyTorch Datasets for audio classification (ROOT-safe)

  AudioDataset          directory layout:  <dir>/positive/*.wav, <dir>/negative/*.wav
  ManifestAudioDataset  CSV manifest:      filepath,label,split[,...]  (from
                        audio/prepare_real_data.py; paths relative to the CSV)
"""
from pathlib import Path
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from preprocess import AudioPreprocessor

# Anchor to project root: .../greenwave
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "audio" / "data" / "processed"


class _MelDatasetBase(Dataset):
    """Shared load->mel->augment->normalize pipeline over self.samples."""

    samples: list[tuple[Path, int]]
    preprocessor: AudioPreprocessor
    augment: bool

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # Load audio (resampled/mono by AudioPreprocessor.load_audio)
        audio = self.preprocessor.load_audio(str(file_path))

        # Extract mel-spectrogram
        mel = self.preprocessor.extract_melspec(audio)

        # Augmentations
        if self.augment:
            if random.random() > 0.5:
                mel = self.preprocessor.spec_augment(
                    mel, freq_mask_param=20, time_mask_param=30
                )
            if random.random() > 0.5:
                gain_db = random.uniform(-6, 6)
                mel = mel * (10 ** (gain_db / 20))

        # Normalize -> tensor (C=1, F, T)
        mel = self.preprocessor.normalize(mel)
        mel_t = torch.from_numpy(mel).unsqueeze(0).float()
        label_t = torch.tensor([label], dtype=torch.float32)
        return mel_t, label_t


class AudioDataset(_MelDatasetBase):
    """Dataset for siren detection (positive/ + negative/ directory layout)"""

    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        augment: bool = False,
        max_samples: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.augment = augment

        # Load file paths and labels
        self.samples: list[tuple[Path, int]] = []

        # Positive class
        pos_dir = self.data_dir / "positive"
        if pos_dir.exists():
            self.samples.extend([(p, 1) for p in pos_dir.glob("*.wav")])

        # Negative class
        neg_dir = self.data_dir / "negative"
        if neg_dir.exists():
            self.samples.extend([(n, 0) for n in neg_dir.glob("*.wav")])

        random.shuffle(self.samples)

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
        n_pos = sum(lbl for _, lbl in self.samples)
        print(f"  Positive: {n_pos}, Negative: {len(self.samples) - n_pos}")


class ManifestAudioDataset(_MelDatasetBase):
    """Dataset driven by a manifest CSV (filepath,label,split,...)."""

    def __init__(
        self,
        manifest_csv: str,
        preprocessor: AudioPreprocessor,
        split: str | None = None,
        augment: bool = False,
        max_samples: int | None = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.preprocessor = preprocessor
        self.augment = augment

        root = self.manifest_csv.parent
        self.samples = []
        with open(self.manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                if split is not None and row.get("split") != split:
                    continue
                self.samples.append((root / row["filepath"], int(row["label"])))

        random.shuffle(self.samples)
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        n_pos = sum(lbl for _, lbl in self.samples)
        print(f"Loaded {len(self.samples)} samples from {self.manifest_csv.name}"
              f" (split={split or 'all'})")
        print(f"  Positive: {n_pos}, Negative: {len(self.samples) - n_pos}")


def collate_fn(batch):
    """
    Pad variable-length spectrograms on the time dimension to max length in batch.
    """
    specs, labels = zip(*batch)
    max_T = max(s.size(2) for s in specs)
    padded = []
    for s in specs:
        if s.size(2) < max_T:
            pad_T = max_T - s.size(2)
            s = F.pad(s, (0, pad_T))  # pad last dim (T)
        padded.append(s)
    specs = torch.stack(padded)
    labels = torch.stack(labels)
    return specs, labels


def test_dataset():
    """Test dataset loading & collation (ROOT-safe paths)."""
    print("=" * 60)
    print("Audio Dataset Test")
    print("=" * 60)

    # Expect processed/train from prepare_data.py
    data_dir = PROC_DIR / "train"
    if not data_dir.exists():
        print(f"[WARN] Data directory not found: {data_dir}")
        print("  Run: python audio\\prepare_data.py")
        return

    # Use explicit config path so it works from anywhere
    pre = AudioPreprocessor(config_path=str(ROOT / "common" / "config.yaml"))

    ds = AudioDataset(str(data_dir), preprocessor=pre, augment=True)
    if len(ds) == 0:
        print("[WARN] No samples found in dataset")
        return

    print("\nTesting sample loading...")
    for i in range(min(3, len(ds))):
        spec, label = ds[i]
        print(f"  Sample {i}: spec={tuple(spec.shape)} label={label.item()}")

    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    batch_specs, batch_labels = next(iter(dl))
    print(f"\nBatch specs shape: {tuple(batch_specs.shape)}")
    print(f"Batch labels shape: {tuple(batch_labels.shape)}")
    print("\n[OK] Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
