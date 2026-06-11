"""Unit tests for audio/prepare_real_data.py -- no network, no real datasets."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))

from audio.prepare_real_data import (   # noqa: E402
    Recording,
    assign_splits,
    build_dataset,
    classify_relpath,
    rms_dbfs,
    slice_windows,
)

SR = 16000


def tone(duration_sec: float, freq: float = 800.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# slice_windows
# ---------------------------------------------------------------------------

def test_slice_windows_count_and_length():
    audio = tone(9.0)  # 9 s -> windows at 0,1.5,3.0,4.5,6.0 = 5 windows of 3 s
    wins = slice_windows(audio, SR, window_sec=3.0, hop_sec=1.5)
    assert len(wins) == 5
    assert all(len(w) == 3 * SR for w in wins)


def test_slice_windows_pads_short_audio():
    audio = tone(1.5)  # >= 1/3 of window -> padded to exactly 3 s
    wins = slice_windows(audio, SR, window_sec=3.0, hop_sec=1.5)
    assert len(wins) == 1
    assert len(wins[0]) == 3 * SR


def test_slice_windows_drops_tiny_audio():
    audio = tone(0.5)  # < 1/3 of window -> dropped
    assert slice_windows(audio, SR, window_sec=3.0, hop_sec=1.5) == []


def test_slice_windows_drops_silence():
    silent = np.zeros(SR * 6, dtype=np.float32)
    assert slice_windows(silent, SR, window_sec=3.0, hop_sec=1.5) == []


def test_slice_windows_respects_max():
    audio = tone(30.0)
    wins = slice_windows(audio, SR, 3.0, 1.5, max_windows=4)
    assert len(wins) == 4


def test_rms_dbfs_sane():
    assert rms_dbfs(np.zeros(100)) <= -100
    loud = rms_dbfs(tone(1.0, amp=0.5))
    quiet = rms_dbfs(tone(1.0, amp=0.005))
    assert loud > quiet


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------

def test_assign_splits_deterministic_and_stratified():
    origins = {1: [f"pos{i}" for i in range(100)],
               0: [f"neg{i}" for i in range(100)]}
    s1 = assign_splits(origins, val_fraction=0.2, seed=7)
    s2 = assign_splits(origins, val_fraction=0.2, seed=7)
    assert s1 == s2  # deterministic

    val_pos = sum(1 for o in origins[1] if s1[o] == "val")
    val_neg = sum(1 for o in origins[0] if s1[o] == "val")
    assert val_pos == 20 and val_neg == 20  # stratified


def test_assign_splits_small_groups_get_one_val():
    s = assign_splits({1: ["a", "b", "c"]}, val_fraction=0.15, seed=1)
    assert sum(1 for v in s.values() if v == "val") == 1


# ---------------------------------------------------------------------------
# classify_relpath
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("Ambulance/sound_1.wav", 1),
    ("sounds/Firetruck/f12.wav", 1),
    ("sireNNet/police/p_3.wav", 1),
    ("Traffic/t_9.wav", 0),
    ("Road_Noises/r1.wav", 0),
    ("dataset/sirens/s5.wav", 1),
    ("Siren_vs_Road/road/x.wav", 0),       # negative dir wins by depth
    ("misc/unknown/u.wav", None),
])
def test_classify_relpath(path, expected):
    assert classify_relpath(path) == expected


# ---------------------------------------------------------------------------
# build_dataset end-to-end on a tiny fake tree (no downloads)
# ---------------------------------------------------------------------------

def test_build_dataset_end_to_end(tmp_path):
    sf = pytest.importorskip("soundfile")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    recordings = []
    # 6 positive + 6 negative fake recordings, 6 s each -> windows
    for i in range(6):
        p = src_dir / f"pos_{i}.wav"
        sf.write(str(p), tone(6.0, freq=900 + 50 * i), SR)
        recordings.append(Recording(p, 1, "fake", f"fake:pos_{i}"))
        n = src_dir / f"neg_{i}.wav"
        sf.write(str(n), tone(6.0, freq=200 + 10 * i, amp=0.2), SR)
        recordings.append(Recording(n, 0, "fake", f"fake:neg_{i}"))

    rd_cfg = {
        "sample_rate": SR, "window_sec": 3.0, "window_hop_sec": 1.5,
        "min_rms_dbfs": -45.0, "val_fraction": 0.2, "seed": 42,
        "max_windows_per_origin": 3, "max_neg_per_pos": 1.5,
    }
    out_root = tmp_path / "real"
    summary = build_dataset(recordings, rd_cfg, out_root)

    manifest = out_root / "manifest.csv"
    assert manifest.exists()
    assert (out_root / "manifest_train.csv").exists()
    assert (out_root / "manifest_val.csv").exists()

    import csv as _csv
    rows = list(_csv.DictReader(open(manifest)))
    assert rows, "manifest is empty"

    # every referenced window exists, is 3 s @ 16 kHz
    for r in rows[:10]:
        w, sr = sf.read(str(out_root / r["filepath"]))
        assert sr == SR and len(w) == 3 * SR

    # no origin appears in both splits (leakage check)
    split_by_origin = {}
    for r in rows:
        prev = split_by_origin.setdefault(r["origin"], r["split"])
        assert prev == r["split"], f"origin {r['origin']} leaked across splits"

    # both splits contain both labels
    for split in ("train", "val"):
        labels = {r["label"] for r in rows if r["split"] == split}
        assert labels == {"0", "1"}

    # per-origin cap respected
    from collections import Counter
    per_origin = Counter(r["origin"] for r in rows)
    assert max(per_origin.values()) <= 3
