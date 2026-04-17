#!/usr/bin/env python3
"""
Test audio preprocessing with synthetic data
"""
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import urllib.request
ROOT = Path(__file__).resolve().parents[1]   # D:/project G/greenwave


from preprocess import AudioPreprocessor, AudioAugmenter


def generate_synthetic_siren(duration=2.0, sr=16000):
    """Generate synthetic siren (for testing)"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Wail pattern: 500-1200 Hz sweep
    freq = 500 + 700 * np.sin(2 * np.pi * 0.5 * t)
    siren = np.sin(2 * np.pi * freq * t)
    
    # Envelope
    envelope = np.exp(-t * 0.3) * 0.5 + 0.5
    siren *= envelope
    
    return siren


def generate_synthetic_noise(duration=2.0, sr=16000):
    """Generate synthetic traffic noise (for testing)"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Pink noise approximation
    noise = np.random.randn(len(t))
    
    # Low-pass filter (traffic is mostly low freq)
    from scipy import signal
    b, a = signal.butter(4, 1000 / (sr / 2))
    noise = signal.filtfilt(b, a, noise)
    
    return noise * 0.3


def test_preprocessing():
    """Test the full preprocessing pipeline"""
    print("=" * 60)
    print("Audio Preprocessing Test")
    print("=" * 60)
    
    # Create test data directory
    test_dir = ROOT / "audio" / "data" / "test"
    (test_dir / "sirens").mkdir(parents=True, exist_ok=True)
    (test_dir / "noise").mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    
    # Generate synthetic audio
    print("\n[1/4] Generating synthetic test audio...")
    siren = generate_synthetic_siren(duration=2.0, sr=sr)
    noise = generate_synthetic_noise(duration=2.0, sr=sr)
    
    sf.write(test_dir / "sirens" / "test_siren.wav", siren, sr)
    sf.write(test_dir / "noise" / "test_noise.wav", noise, sr)
    print("[OK] Saved test audio files")
    
    # Test preprocessor
    print("\n[2/4] Testing mel-spectrogram extraction...")
    preprocessor = AudioPreprocessor(config_path=str(ROOT / "common" / "config.yaml"))
    
    mel_spec = preprocessor.extract_melspec(siren)
    print(f"[OK] Mel-spec shape: {mel_spec.shape} (n_mels={mel_spec.shape[0]}, frames={mel_spec.shape[1]})")
    
    # Test SpecAugment
    print("\n[3/4] Testing SpecAugment...")
    mel_spec_aug = preprocessor.spec_augment(mel_spec)
    print(f"[OK] Augmented spec shape: {mel_spec_aug.shape}")
    
    # Test SNR mixing
    print("\n[4/4] Testing SNR mixing...")
    augmenter = AudioAugmenter(sr)
    
    for snr in [-5, 0, 5, 10, 15]:
        mixed, _ = augmenter.set_snr(siren, noise, snr)
        print(f"  SNR={snr:+3d}dB: RMS={np.sqrt(np.mean(mixed**2)):.4f}")
    
    print("\n[OK] All preprocessing tests passed!")
    
    # Visualize
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original mel-spec
    axes[0, 0].imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('Original Mel-Spectrogram')
    axes[0, 0].set_ylabel('Mel Bin')
    
    # Augmented mel-spec
    axes[0, 1].imshow(mel_spec_aug, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title('SpecAugment Applied')
    
    # Waveforms
    axes[1, 0].plot(siren[:1000])
    axes[1, 0].set_title('Siren Waveform (first 1000 samples)')
    axes[1, 0].set_xlabel('Sample')
    
    mixed, _ = augmenter.set_snr(siren, noise, snr_db=5)
    axes[1, 1].plot(mixed[:1000])
    axes[1, 1].set_title('Mixed at SNR=5dB')
    axes[1, 1].set_xlabel('Sample')
    
    plt.tight_layout()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "preprocess_test.png", dpi=150)
    print(f"[OK] Saved visualization to {out_dir / 'preprocess_test.png'}")



if __name__ == "__main__":
    test_preprocessing()