#!/usr/bin/env python3
"""
Streaming audio detector with bearing estimation (ROOT-safe)
Outputs: p_siren + bearing_deg every hop (default 100 ms),
plus a Doppler factor that discounts receding sirens.
"""
from collections import deque
from pathlib import Path
import numpy as np
import yaml

from infer import SirenDetector
from bearing import MicrophoneArray, BearingEstimator

# Anchor to project root
ROOT = Path(__file__).resolve().parents[1]
CFG_DEFAULT = ROOT / "common" / "config.yaml"
CKPT_DEFAULT = ROOT / "checkpoints" / "audio_best.pt"


class DopplerTracker:
    """
    Decides whether a siren is approaching or receding from its pitch trend.

    Raw siren pitch is useless for this -- a wail sweeps hundreds of Hz per
    second by design. So we take the LOUDEST frequency in each half-second
    block (the top of the sweep, which stays put while the vehicle hangs at
    one distance) and fit a line through those block peaks over the last
    couple of seconds. A clearly falling line means the vehicle is driving
    away, and that detection's belief contribution gets multiplied down.
    """

    def __init__(self, sample_rate: int = 16000, window_sec: float = 2.0,
                 block_sec: float = 0.5, band_hz=(500, 2500),
                 falling_slope_hz_per_s: float = -10.0,
                 receding_factor: float = 0.3):
        self.sr = sample_rate
        self.block_samples = int(block_sec * sample_rate)
        self.band = band_hz
        self.falling_slope = falling_slope_hz_per_s
        self.receding_factor = receding_factor
        self.n_blocks = max(3, int(round(window_sec / block_sec)))

        self._buf = np.empty(0, dtype=np.float32)
        self._buf_t0 = None            # media time of the buffer start
        self._peaks = deque(maxlen=self.n_blocks)   # (block_time, peak_hz)

    def reset(self):
        self._buf = np.empty(0, dtype=np.float32)
        self._buf_t0 = None
        self._peaks.clear()

    def update(self, chunk: np.ndarray, t: float) -> dict:
        """Feed one detected-siren chunk; get the current verdict back."""
        if self._buf_t0 is None:
            self._buf_t0 = t - len(self._buf) / self.sr
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])

        # peel off full blocks
        while len(self._buf) >= self.block_samples:
            block = self._buf[:self.block_samples]
            self._buf = self._buf[self.block_samples:]
            peak = self._block_peak_hz(block)
            if peak is not None:
                self._peaks.append((self._buf_t0, peak))
            self._buf_t0 += self.block_samples / self.sr

        return self.verdict()

    def _block_peak_hz(self, block: np.ndarray):
        """Strongest frequency in the siren band, or None if too quiet."""
        if np.sqrt(np.mean(block ** 2)) < 1e-4:
            return None
        spec = np.abs(np.fft.rfft(block * np.hanning(len(block))))
        freqs = np.fft.rfftfreq(len(block), 1.0 / self.sr)
        lo, hi = self.band
        in_band = (freqs >= lo) & (freqs <= hi)
        if not in_band.any() or spec[in_band].max() <= 0:
            return None
        idx = np.argmax(spec * in_band)
        return float(freqs[idx])

    def verdict(self) -> dict:
        """Factor + slope from the current envelope history."""
        if len(self._peaks) < 3:
            return {"doppler_factor": 1.0, "pitch_slope_hz_s": 0.0}
        ts = np.array([t for t, _ in self._peaks])
        fs = np.array([f for _, f in self._peaks])
        if ts[-1] - ts[0] < 1.0:       # not enough time span to call a trend
            return {"doppler_factor": 1.0, "pitch_slope_hz_s": 0.0}
        slope = float(np.polyfit(ts - ts[0], fs, 1)[0])
        receding = slope < self.falling_slope
        return {
            "doppler_factor": self.receding_factor if receding else 1.0,
            "pitch_slope_hz_s": round(slope, 1),
        }

    @classmethod
    def from_config(cls, config: dict):
        d = config["audio"].get("doppler", {})
        if not d.get("enabled", True):
            return None
        return cls(
            sample_rate=int(config["audio"]["sample_rate"]),
            window_sec=float(d.get("window_sec", 2.0)),
            block_sec=float(d.get("block_sec", 0.5)),
            band_hz=tuple(d.get("band_hz", (500, 2500))),
            falling_slope_hz_per_s=float(d.get("falling_slope_hz_per_s", -10.0)),
            receding_factor=float(d.get("receding_factor", 0.3)),
        )

class StreamingDetector:
    """Unified streaming detector: siren prob + bearing"""

    def __init__(self, model_path: str = str(CKPT_DEFAULT), config_path: str = str(CFG_DEFAULT)):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Siren detector (channel 0)
        self.siren_detector = SirenDetector(model_path=model_path, config_path=config_path)

        # Microphone array
        mic_positions = self.config["audio"]["mic_array"]["positions"]
        self.mic_array = MicrophoneArray(mic_positions)

        # Bearing estimator
        self.bearing_estimator = BearingEstimator(
            self.mic_array,
            sample_rate=self.config["audio"]["sample_rate"],
            median_window=self.config["audio"]["bearing_median_window"],
        )

        # Doppler trend tracker (None when disabled in config)
        self.doppler = DopplerTracker.from_config(self.config)
        self._t_audio = 0.0   # running media time, advanced by chunk length

        print("[OK] Streaming detector initialized")

    def process_multichannel(self, audio_channels: list[np.ndarray]) -> dict:
        """
        Args:
            audio_channels: [ch0, ch1, ch2, ...] each 1D np.ndarray of same length
        Returns:
            dict with timestamp (optional), p_siren, bearing_deg, bearing_confidence,
            detected, tdoas, doppler_factor, pitch_slope_hz_s
        """
        if not audio_channels:
            return {"timestamp": None, "p_siren": 0.0, "bearing_deg": 0.0,
                    "bearing_confidence": 0.0, "detected": False, "tdoas": [],
                    "doppler_factor": 1.0, "pitch_slope_hz_s": 0.0}

        # Siren on first channel
        siren_result = self.siren_detector.process_chunk(audio_channels[0])
        self._t_audio += len(audio_channels[0]) / self.config["audio"]["sample_rate"]

        # Bearing only if siren is currently detected
        if siren_result["detected"] and len(audio_channels) >= 2:
            bearing_result = self.bearing_estimator.estimate_bearing(audio_channels)
        else:
            bearing_result = {"bearing_deg": 0.0, "confidence": 0.0, "tdoas": []}

        # Doppler: only feed the tracker while we actually hear a siren,
        # otherwise traffic noise would pollute the pitch trend
        doppler = {"doppler_factor": 1.0, "pitch_slope_hz_s": 0.0}
        if self.doppler is not None:
            if siren_result["detected"]:
                doppler = self.doppler.update(audio_channels[0], self._t_audio)
            else:
                doppler = self.doppler.verdict()

        return {
            "timestamp": None,
            "p_siren": float(siren_result["p_siren"]),
            "bearing_deg": float(bearing_result["bearing_deg"]),
            "bearing_confidence": float(bearing_result["confidence"]),
            "detected": bool(siren_result["detected"]),
            "tdoas": bearing_result.get("tdoas", []),
            **doppler,
        }

    def reset(self):
        self.siren_detector.reset()
        self.bearing_estimator.reset()
        if self.doppler is not None:
            self.doppler.reset()


def simulate_moving_source():
    """
    Simulate a moving siren source for testing (no mic hardware required).
    """
    print("=" * 60)
    print("Simulated Moving Source Test")
    print("=" * 60)

    model_path = CKPT_DEFAULT
    if not model_path.exists():
        print(f"[WARN] Model not found: {model_path}")
        print("  Proceeding with bearing-only simulation.\n")

        # Bearing-only test
        positions = [[0.0, 0.0], [0.15, 0.0], [0.075, 0.13]]
        array = MicrophoneArray(positions)
        estimator = BearingEstimator(array, sample_rate=16000)

        sr = 16000
        chunk_duration = 0.1
        chunk_samples = int(sr * chunk_duration)
        n_chunks = 20

        print("Simulating source moving from 0° -> 180° over 2s...")
        results = []
        for i in range(n_chunks):
            true_angle = (i / n_chunks) * 180.0
            t = np.linspace(0, chunk_duration, chunk_samples, endpoint=False)
            source = np.sin(2 * np.pi * 1000 * t)

            channels = []
            for mic_pos in array.positions:
                angle_rad = np.radians(true_angle)
                direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                delay = -np.dot(mic_pos, direction) / 343.0
                delay_samples = int(delay * sr)
                if delay_samples > 0:
                    sig = np.pad(source, (delay_samples, 0))[:-delay_samples]
                elif delay_samples < 0:
                    sig = np.pad(source, (0, -delay_samples))[-delay_samples:]
                else:
                    sig = source.copy()
                sig += np.random.randn(len(sig)) * 0.05
                channels.append(sig)

            result = estimator.estimate_bearing(channels)
            results.append({"time": i * chunk_duration, "true_angle": true_angle, "est_angle": result["bearing_deg"], "confidence": result["confidence"]})
            print(f"  t={i*chunk_duration:4.1f}s | True={true_angle:6.1f}°  Est={result['bearing_deg']:6.1f}°  (conf={result['confidence']:.3f})")

        errors = [abs(r["est_angle"] - r["true_angle"]) for r in results]
        mean_error = float(np.mean(errors))
        print(f"\nMean tracking error: {mean_error:.1f}°")
        print("[OK] Simulation complete!")
        return results

    else:
        print("Full detector test (siren + bearing)...")
        det = StreamingDetector(model_path=str(model_path), config_path=str(CFG_DEFAULT))
        print("[OK] Ready for real audio streams (feed your mic channels to process_multichannel()).")
        return None


if __name__ == "__main__":
    simulate_moving_source()
