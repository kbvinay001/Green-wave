#!/usr/bin/env python3
"""
Real-time/file inference for siren detection (ROOT-safe)
- Loads audio_best.pt from checkpoints by default
- Works from any working directory
- CLI: --file path\to\audio.wav  (optional)
"""
from pathlib import Path
from collections import deque
import argparse
import numpy as np
import torch
import yaml

from model import SirenCRNN
from preprocess import AudioPreprocessor

# ----- Paths anchored to project root -----
ROOT = Path(__file__).resolve().parents[1]
CFG_DEFAULT = ROOT / "common" / "config.yaml"
CKPT_DEFAULT = ROOT / "checkpoints" / "audio_best.pt"
TEST_WAV_DEFAULT = ROOT / "audio" / "data" / "test" / "sirens" / "test_siren.wav"


class SirenDetector:
    """Real-time / sliding-window siren detector."""

    def __init__(self, model_path: str = str(CKPT_DEFAULT), config_path: str = str(CFG_DEFAULT), device: str | None = None):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model/checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = SirenCRNN(n_mels=self.config["audio"]["n_mels"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.threshold = float(ckpt.get("threshold", 0.5))

        # Preprocessor
        self.pre = AudioPreprocessor(config_path)

        # Streaming window/hop
        self.sr = int(self.config["audio"]["sample_rate"])
        self.window_sec = float(self.config["audio"]["window_sec"])
        self.hop_sec = float(self.config["audio"]["hop_sec"])
        self.window_samples = int(self.window_sec * self.sr)
        self.hop_samples = int(self.hop_sec * self.sr)

        # Ring buffer
        self.buffer = deque(maxlen=self.window_samples)

        print(f"[OK] Loaded model: {model_path}")
        print(f"  Device: {self.device} | Threshold: {self.threshold:.3f}")

    def _infer_window(self, wav: np.ndarray) -> float:
        """Return p(siren) for a full window of samples."""
        mel = self.pre.extract_melspec(wav)
        mel = self.pre.normalize(mel)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(self.device)  # (1,1,F,T)
        with torch.no_grad():
            p = self.model(x).item()  # SirenCRNN already applies sigmoid
        return float(p)

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Push a new chunk (1D float32) and get current detection."""
        self.buffer.extend(audio_chunk.tolist())
        if len(self.buffer) < self.window_samples:
            return {"p_siren": 0.0, "detected": False, "confidence": 0.0}

        window = np.array(self.buffer, dtype=np.float32)
        p = self._infer_window(window)
        return {"p_siren": p, "detected": p >= self.threshold, "confidence": p}

    def process_file(self, audio_path: str, hop_sec: float | None = None):
        """Slide a window across an audio file, return list of dicts with timestamps."""
        hop = hop_sec if hop_sec is not None else self.hop_sec
        hop_samples = int(hop * self.sr)
        y = self.pre.load_audio(audio_path)  # mono @ target sr

        results = []
        self.buffer.clear()

        for start in range(0, max(0, len(y) - self.window_samples + 1), hop_samples):
            chunk = y[start:start + self.window_samples]
            res = self.process_chunk(chunk)
            res["timestamp"] = start / self.sr
            results.append(res)

        return results

    def reset(self):
        self.buffer.clear()


def cli():
    ap = argparse.ArgumentParser(description="Siren inference (file / streaming)")
    ap.add_argument("--model", default=str(CKPT_DEFAULT), help="Path to checkpoint (audio_best.pt)")
    ap.add_argument("--config", default=str(CFG_DEFAULT), help="Path to config.yaml")
    ap.add_argument("--file", default=str(TEST_WAV_DEFAULT), help="WAV file to run inference on")
    ap.add_argument("--hop", type=float, default=None, help="Hop seconds (override config)")
    args = ap.parse_args()

    m_path = Path(args.model)
    if not m_path.exists():
        print(f"[FAIL] Model not found: {m_path}")
        print("  Train first (python audio\\train.py) or pass --model path")
        return

    f_path = Path(args.file)
    if not f_path.exists():
        print(f"[FAIL] Test audio not found: {f_path}")
        print("  Tip: run python audio\\test_preprocess.py to generate a test siren WAV.")
        return

    det = SirenDetector(model_path=str(m_path), config_path=str(args.config))
    print(f"\nProcessing: {f_path}")
    results = det.process_file(str(f_path), hop_sec=args.hop)

    print(f"Processed {len(results)} windows\nSample results:")
    for r in results[:8]:
        status = "SIREN" if r["detected"] else "noise"
        print(f"  {r['timestamp']:.2f}s: p={r['p_siren']:.4f} [{status}]")

    # quick latency estimate on a few hops
    import time
    y = det.pre.load_audio(str(f_path))
    hop_samples = det.hop_samples
    times = []
    for i in range(min(5, max(1, len(y) // hop_samples - 1))):
        chunk = y[i * hop_samples:(i + 1) * hop_samples]
        t0 = time.time()
        _ = det.process_chunk(chunk)
        times.append((time.time() - t0) * 1000.0)
    if times:
        avg = float(np.mean(times))
        print(f"\nAvg per-chunk latency: {avg:.2f} ms (target < 100 ms: {'[OK]' if avg < 100 else '[FAIL]'})")

if __name__ == "__main__":
    cli()
