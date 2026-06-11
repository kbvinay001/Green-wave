#!/usr/bin/env python3
"""
Real-data ingestion for the siren CRNN -- Green Wave++

Builds a windowed, manifest-driven training set from four free datasets:

  POSITIVES (sirens):
    evss      Emergency Vehicle Siren Sounds (Kaggle vishnu0399)
              Ambulance/, Firetruck/  ->  positive;  Traffic/  ->  negative
    sirennet  sireNNet (Mendeley j4ydzzv4kb v1)
              ambulance/firetruck/police  ->  positive;  traffic  ->  negative
    lssiren   LSSiren (figshare 17560865 -> Google Drive folder)
              siren class  ->  positive;  road-noise class  ->  negative

  HARD NEGATIVES (non-siren urban noise):
    urbansound8k  UrbanSound8K (Zenodo 1203745) -- only the classes listed in
                  config audio.real_data.urbansound_negative_classes

Every accepted recording is resampled to 16 kHz mono and sliced into 3 s
windows (hop 1.5 s).  The train/val split happens at SOURCE-RECORDING level
(grouped by Freesound ID for UrbanSound8K) so windows from one recording can
never leak across splits.

Outputs (under audio/data/real/):
    windows/positive/*.wav      3 s @ 16 kHz mono PCM16
    windows/negative/*.wav
    manifest.csv                filepath,label,split,source,origin
    manifest_train.csv          convenience view (split == train)
    manifest_val.csv            convenience view (split == val)

Usage:
    python audio/prepare_real_data.py --download          # fetch archives only
    python audio/prepare_real_data.py --build             # extract + window + split
    python audio/prepare_real_data.py --download --build  # everything
    python audio/prepare_real_data.py --build --quick     # tiny smoke-test build
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Anchor to greenwave/ root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFG_DEFAULT = ROOT / "common" / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_real_data_config(config_path: Path = CFG_DEFAULT) -> dict:
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    rd = cfg["audio"]["real_data"]
    rd["sample_rate"] = cfg["audio"]["sample_rate"]
    return rd


# ---------------------------------------------------------------------------
# Window slicing (pure -- unit tested)
# ---------------------------------------------------------------------------

def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dB relative to full scale (1.0)."""
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * np.log10(rms)


def slice_windows(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 3.0,
    hop_sec: float = 1.5,
    min_rms_dbfs: float = -45.0,
    max_windows: int | None = None,
) -> list[np.ndarray]:
    """
    Slice a mono recording into fixed-length windows.

    Recordings shorter than one window are zero-padded (centred) if they are
    at least 1/3 of the window long; shorter than that they are dropped.
    Near-silent windows are discarded.
    """
    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))

    if len(audio) < win:
        if len(audio) < win // 3:
            return []
        pad = win - len(audio)
        padded = np.pad(audio, (pad // 2, pad - pad // 2))
        return [padded] if rms_dbfs(padded) >= min_rms_dbfs else []

    out: list[np.ndarray] = []
    for start in range(0, len(audio) - win + 1, hop):
        w = audio[start:start + win]
        if rms_dbfs(w) >= min_rms_dbfs:
            out.append(w.copy())
        if max_windows is not None and len(out) >= max_windows:
            break
    return out


# ---------------------------------------------------------------------------
# Split assignment (pure -- unit tested)
# ---------------------------------------------------------------------------

def assign_splits(
    origins_by_label: dict[int, list[str]],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """
    Deterministic, label-stratified train/val assignment of ORIGIN ids
    (one origin = one source recording).  Returns {origin: "train"|"val"}.
    """
    rng = random.Random(seed)
    split: dict[str, str] = {}
    for label in sorted(origins_by_label):
        origins = sorted(set(origins_by_label[label]))
        rng.shuffle(origins)
        n_val = max(1, round(len(origins) * val_fraction)) if origins else 0
        for i, origin in enumerate(origins):
            split[origin] = "val" if i < n_val else "train"
    return split


# ---------------------------------------------------------------------------
# Path -> class rules (pure -- unit tested)
# ---------------------------------------------------------------------------

_POSITIVE_HINTS = ("ambulance", "firetruck", "fire_truck", "fire-truck", "police", "siren")
_NEGATIVE_HINTS = ("traffic", "road", "noise", "horn")


def classify_relpath(relpath: str) -> int | None:
    """
    Map a file's relative path inside an extracted siren dataset to a label.
    1 = siren, 0 = non-siren, None = unknown (skipped).
    Negative hints win when both appear ("siren_vs_road_noise/road/x.wav").
    """
    p = relpath.lower().replace("\\", "/")
    neg = any(h in p for h in _NEGATIVE_HINTS)
    pos = any(h in p for h in _POSITIVE_HINTS)
    if neg and pos:
        # Decide by the deepest (most specific) matching directory component
        parts = p.split("/")
        for comp in reversed(parts[:-1]):
            if any(h in comp for h in _NEGATIVE_HINTS):
                return 0
            if any(h in comp for h in _POSITIVE_HINTS):
                return 1
        return 0
    if neg:
        return 0
    if pos:
        return 1
    return None


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_all(rd_cfg: dict, downloads_dir: Path) -> None:
    """Fetch every configured archive that isn't already present."""
    downloads_dir.mkdir(parents=True, exist_ok=True)

    for name, src in rd_cfg["sources"].items():
        if "gdrive_folder" in src:
            dest = downloads_dir / src["archive"]
            print(f"[GET ] {name}: Google Drive folder {src['gdrive_folder']}")
            try:
                _gdrive_fetch_folder(src["gdrive_folder"], dest)
            except Exception as e:
                print(f"[WARN] {name}: gdrive fetch failed ({e}) -- continuing without it")
            continue

        dest = downloads_dir / src["archive"]
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"[SKIP] {name}: already downloaded -> {dest.name} "
                  f"({dest.stat().st_size / 1e6:.0f} MB)")
            continue
        print(f"[GET ] {name}: {src['url']}")
        try:
            subprocess.run(
                ["curl", "-L", "-C", "-", "--silent", "--show-error",
                 "-o", str(dest), src["url"]],
                check=True,
            )
            print(f"[OK  ] {name}: {dest.stat().st_size / 1e6:.0f} MB")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] {name}: download failed ({e}) -- continuing without it")


def _gdrive_fetch_folder(folder_id: str, dest: Path) -> None:
    """
    Download a public Google Drive folder, file by file.

    gdown's --folder mode caps at 50 files per folder, so we list the tree
    first (skip_download=True) and then fetch whatever is missing locally.
    Already-present files are skipped, making this resumable across runs
    and across Drive's daily quota resets.
    """
    import time
    import gdown

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    listing = gdown.download_folder(url, output=str(dest),
                                    skip_download=True, quiet=True)
    if not listing:
        raise RuntimeError("folder listing failed (Drive quota or permissions)")

    missing = [f for f in listing if not Path(f.local_path).exists()]
    print(f"  {len(listing)} files in folder, {len(missing)} to download")

    failures = 0
    for i, f in enumerate(missing):
        Path(f.local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(id=f.id, output=str(f.local_path), quiet=True)
        except Exception:
            failures += 1
            if failures > 30:
                print("  [WARN] too many Drive failures -- stopping early "
                      "(rerun --download later to resume)")
                break
            time.sleep(2.0)
        if i and i % 200 == 0:
            print(f"  downloaded {i}/{len(missing)}")
    got = sum(1 for f in listing if Path(f.local_path).exists())
    print(f"  [OK] {got}/{len(listing)} files present locally")


def extract_archive(archive: Path, dest: Path) -> bool:
    """Extract zip/tar.gz; recursively extract one level of nested zips."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        elif archive.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive, "r:gz") as t:
                t.extractall(dest, filter="data")
        else:
            return False
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as e:
        print(f"[WARN] cannot extract {archive.name}: {e}")
        # Remove partial extraction so the next run doesn't mistake it for done
        shutil.rmtree(dest, ignore_errors=True)
        return False

    # One level of nested archives (LSSiren ships zips inside the folder)
    for nested in list(dest.rglob("*.zip")):
        sub = nested.with_suffix("")
        if not sub.exists():
            try:
                with zipfile.ZipFile(nested) as z:
                    z.extractall(sub)
            except (zipfile.BadZipFile, EOFError) as e:
                print(f"[WARN] nested zip {nested.name}: {e}")
    return True


# ---------------------------------------------------------------------------
# Harvest: extracted tree -> labelled recordings
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    path: Path     # absolute path to the audio file
    label: int     # 1 siren, 0 non-siren
    source: str    # evss / sirennet / lssiren / urbansound8k
    origin: str    # split-group key (one per source recording)


_AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".aif", ".aiff")


def harvest_siren_source(source: str, extracted: Path) -> list[Recording]:
    """EVSS / sireNNet / LSSiren: label every audio file by its path."""
    recs: list[Recording] = []
    skipped = 0
    for f in sorted(extracted.rglob("*")):
        if f.suffix.lower() not in _AUDIO_EXTS or not f.is_file():
            continue
        label = classify_relpath(str(f.relative_to(extracted)))
        if label is None:
            skipped += 1
            continue
        origin = f"{source}:{sanitize(f.stem)}"
        recs.append(Recording(f, label, source, origin))
    if skipped:
        print(f"  [{source}] skipped {skipped} files with unrecognised class path")
    return recs


def harvest_urbansound(extracted: Path, wanted_classes: list[str]) -> list[Recording]:
    """UrbanSound8K via its metadata CSV; origin grouped by Freesound ID."""
    meta = next(extracted.rglob("UrbanSound8K.csv"), None)
    if meta is None:
        print("  [urbansound8k] metadata CSV not found -- skipping source")
        return []

    audio_root = meta.parent.parent / "audio"
    recs: list[Recording] = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            if row["class"] not in wanted_classes:
                continue
            p = audio_root / f"fold{row['fold']}" / row["slice_file_name"]
            if not p.exists():
                continue
            recs.append(Recording(p, 0, "urbansound8k", f"us8k:{row['fsID']}"))
    return recs


# ---------------------------------------------------------------------------
# Build: recordings -> windows + manifest
# ---------------------------------------------------------------------------

def build_dataset(
    recordings: list[Recording],
    rd_cfg: dict,
    out_root: Path,
    quick: bool = False,
) -> dict:
    """Window every recording, balance classes, split, write wavs + manifest."""
    import soundfile as sf
    import librosa

    sr          = int(rd_cfg["sample_rate"])
    window_sec  = float(rd_cfg["window_sec"])
    hop_sec     = float(rd_cfg["window_hop_sec"])
    min_rms     = float(rd_cfg["min_rms_dbfs"])
    per_origin  = int(rd_cfg["max_windows_per_origin"])
    seed        = int(rd_cfg["seed"])

    if quick:
        rng = random.Random(seed)
        by_src: dict[tuple, list[Recording]] = defaultdict(list)
        for r in recordings:
            by_src[(r.source, r.label)].append(r)
        recordings = []
        for key, group in sorted(by_src.items()):
            rng.shuffle(group)
            recordings.extend(group[:10])
        print(f"[QUICK] limited to {len(recordings)} recordings")

    # 1. Source-recording-level split (no window leakage)
    origins_by_label: dict[int, list[str]] = defaultdict(list)
    for r in recordings:
        origins_by_label[r.label].append(r.origin)
    split_of = assign_splits(origins_by_label, float(rd_cfg["val_fraction"]), seed)

    # 2. Window + write
    win_dirs = {1: out_root / "windows" / "positive",
                0: out_root / "windows" / "negative"}
    for d in win_dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rows: list[dict] = []
    n_windows_by_origin: dict[str, int] = defaultdict(int)
    errors = 0

    for i, rec in enumerate(recordings):
        if i and i % 500 == 0:
            print(f"  windowed {i}/{len(recordings)} recordings "
                  f"({len(rows)} windows so far)")
        budget = per_origin - n_windows_by_origin[rec.origin]
        if budget <= 0:
            continue
        try:
            audio, _ = librosa.load(str(rec.path), sr=sr, mono=True)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [WARN] unreadable: {rec.path.name} ({e})")
            continue

        windows = slice_windows(audio, sr, window_sec, hop_sec, min_rms,
                                max_windows=budget)
        n_windows_by_origin[rec.origin] += len(windows)

        base = sanitize(f"{rec.source}_{rec.path.stem}")
        for w_idx, w in enumerate(windows):
            cls = "positive" if rec.label else "negative"
            name = f"{base}_{w_idx:02d}.wav"
            path = win_dirs[rec.label] / name
            sf.write(str(path), w.astype(np.float32), sr, subtype="PCM_16")
            rows.append({
                "filepath": str(path.relative_to(out_root)).replace("\\", "/"),
                "label":    rec.label,
                "split":    split_of[rec.origin],
                "source":   rec.source,
                "origin":   rec.origin,
            })

    if errors:
        print(f"  [WARN] {errors} unreadable files skipped")

    # 3. Cap negative:positive ratio (drop excess negatives, deterministic)
    rows = _balance(rows, float(rd_cfg["max_neg_per_pos"]), seed, win_dirs[0], out_root)

    # 4. Manifests
    out_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(out_root / "manifest.csv", rows)
    _write_manifest(out_root / "manifest_train.csv",
                    [r for r in rows if r["split"] == "train"])
    _write_manifest(out_root / "manifest_val.csv",
                    [r for r in rows if r["split"] == "val"])

    return _summarize(rows)


def _balance(rows: list[dict], max_neg_per_pos: float, seed: int,
             neg_dir: Path, out_root: Path) -> list[dict]:
    """Per split, keep at most max_neg_per_pos negatives per positive."""
    rng = random.Random(seed + 1)
    kept: list[dict] = [r for r in rows if r["label"] == 1]
    for split in ("train", "val"):
        pos = [r for r in rows if r["label"] == 1 and r["split"] == split]
        neg = [r for r in rows if r["label"] == 0 and r["split"] == split]
        limit = int(len(pos) * max_neg_per_pos)
        if len(neg) > limit:
            # Sample evenly across sources so one dataset doesn't dominate
            by_source: dict[str, list[dict]] = defaultdict(list)
            for r in neg:
                by_source[r["source"]].append(r)
            chosen: list[dict] = []
            sources = sorted(by_source)
            while len(chosen) < limit and any(by_source[s] for s in sources):
                for s in sources:
                    if by_source[s] and len(chosen) < limit:
                        chosen.append(by_source[s].pop(
                            rng.randrange(len(by_source[s]))))
            dropped = set(id(r) for r in neg) - set(id(r) for r in chosen)
            for r in neg:
                if id(r) in dropped:
                    (out_root / r["filepath"]).unlink(missing_ok=True)
            neg = chosen
            print(f"  [{split}] capped negatives at {limit} "
                  f"({len(pos)} positives, ratio {max_neg_per_pos})")
        kept.extend(neg)
    return kept


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "label", "split",
                                          "source", "origin"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK ] wrote {path.name}  ({len(rows)} rows)")


def _summarize(rows: list[dict]) -> dict:
    summary: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        summary[r["split"]][f"label_{r['label']}"] += 1
        summary[r["split"]][r["source"]] += 1
    return {k: dict(v) for k, v in summary.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build real siren training data")
    ap.add_argument("--download", action="store_true", help="fetch archives")
    ap.add_argument("--build",    action="store_true", help="extract + window + split")
    ap.add_argument("--quick",    action="store_true",
                    help="limit to ~10 recordings per source/class (smoke test)")
    ap.add_argument("--config",   default=str(CFG_DEFAULT))
    args = ap.parse_args()

    if not args.download and not args.build:
        ap.error("nothing to do: pass --download and/or --build")

    rd_cfg = load_real_data_config(Path(args.config))
    out_root = ROOT / rd_cfg["root"]
    downloads = out_root / "downloads"
    extracted_root = out_root / "extracted"

    if args.download:
        download_all(rd_cfg, downloads)

    if not args.build:
        return

    # Extract whatever archives are present
    print("\n[1/3] Extracting archives...")
    extracted_dirs: dict[str, Path] = {}
    for name, src in rd_cfg["sources"].items():
        target = extracted_root / name
        archive = downloads / src["archive"]
        if target.exists() and any(target.rglob("*")):
            print(f"[SKIP] {name}: already extracted")
            extracted_dirs[name] = target
            continue
        if "gdrive_folder" in src:
            if archive.exists() and any(archive.rglob("*")):
                # gdown wrote a plain folder; treat it as extracted, but still
                # unpack any zips inside it
                extract_archive_nested_only(archive)
                extracted_dirs[name] = archive
            else:
                print(f"[MISS] {name}: not downloaded -- skipping")
            continue
        if not archive.exists() or archive.stat().st_size < 1_000_000:
            print(f"[MISS] {name}: archive not present -- skipping")
            continue
        print(f"[EXT ] {name} ...")
        if extract_archive(archive, target):
            extracted_dirs[name] = target

    # Harvest recordings
    print("\n[2/3] Harvesting recordings...")
    recordings: list[Recording] = []
    for name, folder in extracted_dirs.items():
        if name == "urbansound8k":
            recs = harvest_urbansound(folder,
                                      rd_cfg["urbansound_negative_classes"])
        else:
            recs = harvest_siren_source(name, folder)
        n_pos = sum(r.label for r in recs)
        print(f"  {name:14s} {len(recs):5d} recordings "
              f"(pos={n_pos}, neg={len(recs) - n_pos})")
        recordings.extend(recs)

    if not recordings:
        print("[FAIL] no recordings harvested -- check downloads")
        sys.exit(1)

    # Window + split + manifest
    print("\n[3/3] Windowing + splitting...")
    summary = build_dataset(recordings, rd_cfg, out_root, quick=args.quick)

    print("\n" + "=" * 60)
    print("Dataset summary (windows):")
    for split, counts in sorted(summary.items()):
        pos = counts.get("label_1", 0)
        neg = counts.get("label_0", 0)
        srcs = {k: v for k, v in counts.items() if not k.startswith("label_")}
        print(f"  {split:6s}  pos={pos:6d}  neg={neg:6d}   {srcs}")
    print(f"\n[OK ] manifests written under {out_root}")
    print("Next: python audio/train.py --real")


def extract_archive_nested_only(folder: Path) -> None:
    """Unpack zips that live inside an already-downloaded plain folder."""
    for nested in list(folder.rglob("*.zip")):
        sub = nested.with_suffix("")
        if not sub.exists():
            try:
                with zipfile.ZipFile(nested) as z:
                    z.extractall(sub)
                print(f"  [EXT] nested {nested.name}")
            except (zipfile.BadZipFile, EOFError) as e:
                print(f"  [WARN] nested zip {nested.name}: {e}")


if __name__ == "__main__":
    main()
