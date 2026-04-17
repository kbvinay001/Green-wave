#!/usr/bin/env python3
"""
Streaming audio detector with bearing estimation (ROOT-safe)
Outputs: p_siren + bearing_deg every hop (default 100 ms)
"""
from pathlib import Path
import numpy as np
import yaml

from infer import SirenDetector
from bearing import MicrophoneArray, BearingEstimator

# Anchor to project root
ROOT = Path(__file__).resolve().parents[1]
CFG_DEFAULT = ROOT / "common" / "config.yaml"
CKPT_DEFAULT = ROOT / "checkpoints" / "audio_best.pt"

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

        print("[OK] Streaming detector initialized")

    def process_multichannel(self, audio_channels: list[np.ndarray]) -> dict:
        """
        Args:
            audio_channels: [ch0, ch1, ch2, ...] each 1D np.ndarray of same length
        Returns:
            dict with timestamp (optional), p_siren, bearing_deg, bearing_confidence, detected, tdoas
        """
        if not audio_channels:
            return {"timestamp": None, "p_siren": 0.0, "bearing_deg": 0.0, "bearing_confidence": 0.0, "detected": False, "tdoas": []}

        # Siren on first channel
        siren_result = self.siren_detector.process_chunk(audio_channels[0])

        # Bearing only if siren is currently detected
        if siren_result["detected"] and len(audio_channels) >= 2:
            bearing_result = self.bearing_estimator.estimate_bearing(audio_channels)
        else:
            bearing_result = {"bearing_deg": 0.0, "confidence": 0.0, "tdoas": []}

        return {
            "timestamp": None,
            "p_siren": float(siren_result["p_siren"]),
            "bearing_deg": float(bearing_result["bearing_deg"]),
            "bearing_confidence": float(bearing_result["confidence"]),
            "detected": bool(siren_result["detected"]),
            "tdoas": bearing_result.get("tdoas", []),
        }

    def reset(self):
        self.siren_detector.reset()
        self.bearing_estimator.reset()


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
