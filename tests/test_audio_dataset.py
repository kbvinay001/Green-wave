"""Tests for audio/dataset.py manifest-driven dataset."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audio"))

from dataset import ManifestAudioDataset, collate_fn  # noqa: E402
from preprocess import AudioPreprocessor              # noqa: E402

SR = 16000
CFG = str(ROOT / "common" / "config.yaml")


@pytest.fixture(scope="module")
def manifest_dir(tmp_path_factory):
    sf = pytest.importorskip("soundfile")
    root = tmp_path_factory.mktemp("real")
    (root / "windows").mkdir()
    rows = []
    for i in range(6):
        label = i % 2
        split = "val" if i >= 4 else "train"
        rel = f"windows/clip_{i}.wav"
        t = np.linspace(0, 3.0, 3 * SR, endpoint=False)
        sf.write(str(root / rel),
                 (0.3 * np.sin(2 * np.pi * (300 + 100 * i) * t)).astype(np.float32),
                 SR)
        rows.append({"filepath": rel, "label": label, "split": split,
                     "source": "fake", "origin": f"fake:{i}"})
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def test_manifest_dataset_split_filter(manifest_dir):
    pre = AudioPreprocessor(CFG)
    train = ManifestAudioDataset(str(manifest_dir / "manifest.csv"), pre, split="train")
    val = ManifestAudioDataset(str(manifest_dir / "manifest.csv"), pre, split="val")
    assert len(train) == 4
    assert len(val) == 2


def test_manifest_dataset_item_shapes(manifest_dir):
    pre = AudioPreprocessor(CFG)
    ds = ManifestAudioDataset(str(manifest_dir / "manifest.csv"), pre, split="train")
    spec, label = ds[0]
    assert spec.dim() == 3 and spec.size(0) == 1      # (1, n_mels, T)
    assert spec.size(1) == 128
    assert label.item() in (0.0, 1.0)

    # batches collate across (equal) lengths
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    specs, labels = next(iter(dl))
    assert specs.shape[0] == 4 and labels.shape == (4, 1)
    assert torch.isfinite(specs).all()
