#!/usr/bin/env python3
"""
Audio preprocessing for Green Wave++
Log-mel spectrogram extraction with SpecAugment
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import yaml


class AudioPreprocessor:
    """Extract log-mel spectrograms from audio"""

    def __init__(self, config_path: str = "D:/project G/greenwave/common/config.yaml"):

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.cfg = config['audio']
        self.sr = self.cfg['sample_rate']
        self.n_mels = self.cfg['n_mels']
        self.n_fft = self.cfg['n_fft']
        self.win_length = self.cfg['win_length']
        self.hop_length = self.cfg['hop_length']
        self.fmin = self.cfg['fmin']
        self.fmax = self.cfg['fmax']
        
    def load_audio(self, path: str) -> np.ndarray:
        """Load audio file and resample to target SR"""
        audio, sr = librosa.load(path, sr=self.sr, mono=True)
        return audio
    
    def extract_melspec(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel spectrogram
        Returns: (n_mels, time_frames)
        """
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        return log_mel
    
    def spec_augment(
        self, 
        spec: np.ndarray,
        freq_mask_param: int = 20,
        time_mask_param: int = 30,
        n_freq_masks: int = 1,
        n_time_masks: int = 1
    ) -> np.ndarray:
        """
        Apply SpecAugment (masking)
        spec: (n_mels, time_frames)
        """
        spec = spec.copy()
        n_mels, n_frames = spec.shape
        
        # Frequency masking
        for _ in range(n_freq_masks):
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, n_mels - f)
            spec[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(n_time_masks):
            t = np.random.randint(0, min(time_mask_param, n_frames))
            t0 = np.random.randint(0, n_frames - t)
            spec[:, t0:t0+t] = 0
        
        return spec
    
    def normalize(self, spec: np.ndarray) -> np.ndarray:
        """Normalize spectrogram to zero mean, unit variance"""
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + 1e-8)


class AudioAugmenter:
    """Mix sirens with noise at various SNRs"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
    
    def set_snr(
        self,
        signal: np.ndarray,
        noise: np.ndarray,
        snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix signal and noise at target SNR
        Returns: (mixed, noise_adjusted)
        """
        # Calculate signal power
        signal_power = np.mean(signal ** 2)
        
        # Calculate noise power for target SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Adjust noise to target power
        current_noise_power = np.mean(noise ** 2)
        noise_adjusted = noise * np.sqrt(noise_power / (current_noise_power + 1e-8))
        
        # Mix
        mixed = signal + noise_adjusted
        
        # Prevent clipping
        max_val = np.abs(mixed).max()
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)
            
        return mixed, noise_adjusted
    
    def random_segment(self, audio: np.ndarray, duration_samples: int) -> np.ndarray:
        """Extract random segment from audio"""
        if len(audio) <= duration_samples:
            # Pad if too short
            pad_length = duration_samples - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')
        else:
            # Random crop
            start_idx = np.random.randint(0, len(audio) - duration_samples)
            return audio[start_idx:start_idx + duration_samples]
    
    def create_mixed_dataset(
        self,
        siren_dir: Path,
        noise_dir: Path,
        output_dir: Path,
        snr_levels: list = [-5, 0, 5, 10, 15],
        n_samples_per_snr: int = 100,
        duration_sec: float = 2.0
    ):
        """
        Generate mixed audio dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        siren_files = list(Path(siren_dir).glob("*.wav"))
        noise_files = list(Path(noise_dir).glob("*.wav"))
        
        if not siren_files or not noise_files:
            raise ValueError("No audio files found in siren_dir or noise_dir")
        
        duration_samples = int(duration_sec * self.sr)
        
        print(f"Generating {len(snr_levels) * n_samples_per_snr} mixed samples...")
        
        idx = 0
        for snr in snr_levels:
            for i in range(n_samples_per_snr):
                # Random siren and noise
                siren_file = np.random.choice(siren_files)
                noise_file = np.random.choice(noise_files)
                
                # Load
                siren, _ = librosa.load(siren_file, sr=self.sr, mono=True)
                noise, _ = librosa.load(noise_file, sr=self.sr, mono=True)
                
                # Extract segments
                siren_seg = self.random_segment(siren, duration_samples)
                noise_seg = self.random_segment(noise, duration_samples)
                
                # Mix at target SNR
                mixed, _ = self.set_snr(siren_seg, noise_seg, snr)
                
                # Save
                output_path = output_dir / f"mixed_snr{snr:+03d}_{idx:04d}.wav"
                sf.write(output_path, mixed, self.sr)
                
                idx += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"  Generated {idx + 1} samples...")
        
        print(f"[OK] Created {idx} mixed audio files in {output_dir}")


def verify_audio_file(path: str, target_sr: int = 16000) -> bool:
    """Check if audio file meets requirements"""
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        
        # Check sample rate
        if sr != target_sr:
            print(f"[WARN] {path}: Sample rate {sr} != {target_sr}")
            return False
        
        # Check for clipping
        if np.abs(audio).max() > 0.99:
            print(f"[WARN] {path}: Clipping detected")
            return False
        
        # Check for silence
        energy = np.mean(audio ** 2)
        if energy < 1e-6:
            print(f"[WARN] {path}: Silent or very low energy")
            return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] {path}: Error loading - {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Audio Preprocessor Module")
    print("=" * 60)
    
    # Test preprocessor
    preprocessor = AudioPreprocessor()
    print(f"[OK] Initialized with SR={preprocessor.sr}, n_mels={preprocessor.n_mels}")
    
    # Test augmenter
    augmenter = AudioAugmenter()
    print(f"[OK] Augmenter ready")
    
    print("\nTo use:")
    print("1. Place siren WAVs in audio/data/raw/sirens/")
    print("2. Place noise WAVs in audio/data/raw/noise/")
    print("3. Run data preparation script")