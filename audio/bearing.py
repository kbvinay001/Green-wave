#!/usr/bin/env python3
"""
Bearing estimation using GCC-PHAT TDOA
Estimates direction of arrival from microphone array
"""
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from collections import deque
from typing import Tuple, List, Optional
import yaml


class MicrophoneArray:
    """Microphone array configuration"""
    
    def __init__(self, positions: List[List[float]]):
        """
        Args:
            positions: List of [x, y] coordinates in meters
                      e.g., [[0, 0], [0.15, 0], [0.075, 0.13]] for triangle
        """
        self.positions = np.array(positions)
        self.n_mics = len(positions)
        
        # Pre-compute mic pairs and baselines
        self.pairs = []
        self.baselines = []
        
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                self.pairs.append((i, j))
                
                # Baseline vector
                baseline = self.positions[j] - self.positions[i]
                self.baselines.append(baseline)
        
        self.baselines = np.array(self.baselines)
        
        print(f"[OK] Microphone array: {self.n_mics} mics, {len(self.pairs)} pairs")
        for i, (mic1, mic2) in enumerate(self.pairs):
            dist = np.linalg.norm(self.baselines[i])
            print(f"  Pair {mic1}-{mic2}: {dist*100:.1f} cm baseline")
    
    def get_baseline_vector(self, pair_idx: int) -> np.ndarray:
        """Get baseline vector for a mic pair"""
        return self.baselines[pair_idx]
    
    def get_pair(self, pair_idx: int) -> Tuple[int, int]:
        """Get microphone indices for a pair"""
        return self.pairs[pair_idx]


class GCCPHATEstimator:
    """
    GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
    for TDOA estimation
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 1024,
        speed_of_sound: float = 343.0  # m/s at 20°C
    ):
        self.sr = sample_rate
        self.frame_length = frame_length
        self.c = speed_of_sound
        
    def gcc_phat(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute TDOA between two signals using GCC-PHAT
        
        Args:
            sig1, sig2: Audio signals (same length)
        
        Returns:
            tdoa: Time difference in seconds (positive if sig2 leads)
            confidence: Cross-correlation peak value (0-1)
        """
        # Ensure same length
        n = min(len(sig1), len(sig2))
        sig1 = sig1[:n]
        sig2 = sig2[:n]
        
        # Zero-pad to next power of 2 for efficiency
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # FFT
        X1 = fft(sig1, n=n_fft)
        X2 = fft(sig2, n=n_fft)
        
        # Cross-power spectrum
        R = X1 * np.conj(X2)
        
        # Phase transform (PHAT weighting)
        R_phat = R / (np.abs(R) + 1e-10)
        
        # Inverse FFT to get cross-correlation
        cc = np.real(ifft(R_phat))
        
        # Find peak (use fftshift to center zero-lag)
        cc = np.fft.fftshift(cc)
        
        # Peak location
        max_idx = np.argmax(np.abs(cc))
        
        # Convert to lag in samples (centered around n_fft/2)
        lag_samples = max_idx - n_fft // 2
        
        # Convert to time
        tdoa = lag_samples / self.sr
        
        # Confidence (normalized peak height)
        confidence = np.abs(cc[max_idx]) / len(cc)
        
        return tdoa, confidence
    
    def estimate_tdoa_robust(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        max_tdoa: float = 0.001  # 1ms max
    ) -> Tuple[float, float]:
        """
        Robust TDOA estimation with outlier rejection
        """
        tdoa, confidence = self.gcc_phat(sig1, sig2)
        
        # Clamp to physical limits
        if np.abs(tdoa) > max_tdoa:
            tdoa = np.sign(tdoa) * max_tdoa
            confidence *= 0.5  # Reduce confidence for clamped values
        
        return tdoa, confidence


class BearingEstimator:
    """
    Estimate bearing (azimuth angle) from TDOA measurements
    """
    
    def __init__(
        self,
        mic_array: MicrophoneArray,
        sample_rate: int = 16000,
        speed_of_sound: float = 343.0,
        median_window: int = 5
    ):
        self.array = mic_array
        self.sr = sample_rate
        self.c = speed_of_sound
        
        self.gcc_phat = GCCPHATEstimator(sample_rate, speed_of_sound=speed_of_sound)
        
        # Median filter for temporal smoothing
        self.median_window = median_window
        self.bearing_history = deque(maxlen=median_window)
        
    def tdoa_to_bearing(
        self,
        tdoas: List[float],
        confidences: List[float]
    ) -> Tuple[float, float]:
        """
        Convert TDOA measurements to bearing angle
        
        Args:
            tdoas: List of TDOA values for each mic pair
            confidences: Confidence for each TDOA
        
        Returns:
            bearing_deg: Azimuth angle in degrees (0° = forward, 90° = right)
            confidence: Overall confidence (0-1)
        """
        if len(tdoas) == 0:
            return 0.0, 0.0
        
        # Weighted least squares approach
        # For each pair, compute possible bearings
        bearings = []
        weights = []
        
        for pair_idx, (tdoa, conf) in enumerate(zip(tdoas, confidences)):
            baseline = self.array.get_baseline_vector(pair_idx)
            baseline_len = np.linalg.norm(baseline)
            
            # Distance difference = c * tdoa
            dist_diff = self.c * tdoa
            
            # Clamp to physical limits (±baseline length)
            dist_diff = np.clip(dist_diff, -baseline_len, baseline_len)
            
            # Angle from baseline axis
            # cos(angle) = dist_diff / baseline_len
            cos_angle = dist_diff / (baseline_len + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Angle relative to baseline
            angle_rel = np.arccos(cos_angle)
            
            # Convert to global bearing (assume baseline along x-axis for simplicity)
            # This is simplified - real implementation needs proper coordinate transform
            baseline_angle = np.arctan2(baseline[1], baseline[0])
            
            if tdoa > 0:
                bearing = baseline_angle + angle_rel
            else:
                bearing = baseline_angle - angle_rel
            
            bearings.append(bearing)
            weights.append(conf)
        
        # Weighted average
        bearings = np.array(bearings)
        weights = np.array(weights)
        
        if weights.sum() < 1e-6:
            return 0.0, 0.0
        
        # Handle angle wrapping
        # Convert to unit vectors and average
        x = np.sum(weights * np.cos(bearings)) / weights.sum()
        y = np.sum(weights * np.sin(bearings)) / weights.sum()
        
        bearing_rad = np.arctan2(y, x)
        bearing_deg = np.degrees(bearing_rad)
        
        # Normalize to [0, 360)
        if bearing_deg < 0:
            bearing_deg += 360
        
        # Overall confidence
        confidence = np.mean(weights)
        
        return bearing_deg, confidence
    
    def estimate_bearing(
        self,
        audio_channels: List[np.ndarray],
        use_median: bool = True
    ) -> dict:
        """
        Estimate bearing from multi-channel audio
        
        Args:
            audio_channels: List of audio arrays, one per microphone
            use_median: Apply median filtering over time
        
        Returns:
            dict with 'bearing_deg', 'confidence', 'tdoas'
        """
        if len(audio_channels) != self.array.n_mics:
            raise ValueError(f"Expected {self.array.n_mics} channels, got {len(audio_channels)}")
        
        # Compute TDOA for each pair
        tdoas = []
        confidences = []
        
        for pair_idx, (mic1, mic2) in enumerate(self.array.pairs):
            sig1 = audio_channels[mic1]
            sig2 = audio_channels[mic2]
            
            # Estimate TDOA
            tdoa, conf = self.gcc_phat.estimate_tdoa_robust(sig1, sig2)
            
            tdoas.append(tdoa)
            confidences.append(conf)
        
        # Convert to bearing
        bearing_deg, confidence = self.tdoa_to_bearing(tdoas, confidences)
        
        # Temporal smoothing with median filter
        if use_median:
            self.bearing_history.append(bearing_deg)
            
            if len(self.bearing_history) >= self.median_window:
                # Handle angle wrapping for median
                bearings_array = np.array(self.bearing_history)
                
                # Convert to complex representation for circular median
                angles_complex = np.exp(1j * np.radians(bearings_array))
                median_complex = np.median(angles_complex.real) + 1j * np.median(angles_complex.imag)
                
                bearing_deg = np.degrees(np.angle(median_complex))
                if bearing_deg < 0:
                    bearing_deg += 360
        
        return {
            'bearing_deg': bearing_deg,
            'confidence': confidence,
            'tdoas': tdoas,
            'tdoa_confidences': confidences
        }
    
    def reset(self):
        """Reset temporal filter"""
        self.bearing_history.clear()


def test_bearing_estimation():
    """Test bearing estimation with synthetic data"""
    print("=" * 60)
    print("Bearing Estimation Test")
    print("=" * 60)
    
    # Create triangular array (15cm baseline)
    positions = [
        [0.0, 0.0],      # Mic 0
        [0.15, 0.0],     # Mic 1
        [0.075, 0.13]    # Mic 2 (equilateral triangle)
    ]
    
    array = MicrophoneArray(positions)
    estimator = BearingEstimator(array, sample_rate=16000)
    
    # Test with synthetic signal
    print("\n[1/3] Testing with synthetic signals...")
    
    sr = 16000
    duration = 0.5  # 500ms
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create source signal (1 kHz tone)
    source = np.sin(2 * np.pi * 1000 * t)
    
    # Test different source directions
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    print("\nTesting different source angles:")
    for true_angle in test_angles:
        # Simulate arrival at each mic
        channels = []
        
        for mic_pos in array.positions:
            # Simplified: assume far-field source
            angle_rad = np.radians(true_angle)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            # Time delay for this mic
            delay = -np.dot(mic_pos, direction) / 343.0  # speed of sound
            delay_samples = int(delay * sr)
            
            # Apply delay
            if delay_samples > 0:
                sig = np.pad(source, (delay_samples, 0))[:-delay_samples]
            elif delay_samples < 0:
                sig = np.pad(source, (0, -delay_samples))[-delay_samples:]
            else:
                sig = source.copy()
            
            # Add small noise
            sig += np.random.randn(len(sig)) * 0.01
            
            channels.append(sig)
        
        # Estimate bearing
        result = estimator.estimate_bearing(channels, use_median=False)
        
        error = abs(result['bearing_deg'] - true_angle)
        if error > 180:
            error = 360 - error
        
        status = "[OK]" if error < 30 else "[FAIL]"
        print(f"  True: {true_angle:3.0f}° -> Est: {result['bearing_deg']:6.1f}° "
              f"(error: {error:4.1f}°, conf: {result['confidence']:.3f}) {status}")
    
    # Test TDOA directly
    print("\n[2/3] Testing GCC-PHAT TDOA...")
    gcc = GCCPHATEstimator(sample_rate=sr)
    
    # Create signals with known delay
    sig1 = np.sin(2 * np.pi * 1000 * t)
    true_delay = 0.0005  # 0.5ms
    delay_samples = int(true_delay * sr)
    sig2 = np.roll(sig1, delay_samples)
    
    tdoa, conf = gcc.gcc_phat(sig1, sig2)
    
    print(f"True delay: {true_delay*1000:.3f} ms")
    print(f"Estimated:  {tdoa*1000:.3f} ms")
    print(f"Error:      {abs(tdoa - true_delay)*1000:.3f} ms")
    print(f"Confidence: {conf:.4f}")
    
    # Test median filtering
    print("\n[3/3] Testing temporal smoothing...")
    estimator.reset()
    
    # Simulate noisy measurements
    true_bearing = 90.0
    noisy_bearings = true_bearing + np.random.randn(10) * 10  # ±10° noise
    
    results = []
    for bearing in noisy_bearings:
        estimator.bearing_history.append(bearing)
        
        if len(estimator.bearing_history) >= 5:
            bearings_array = np.array(estimator.bearing_history)
            smoothed = np.median(bearings_array)
            results.append(smoothed)
    
    print(f"Input noise std: {np.std(noisy_bearings):.1f}°")
    print(f"After median filter: {np.std(results):.1f}°")
    print(f"Noise reduction: {(1 - np.std(results)/np.std(noisy_bearings))*100:.1f}%")
    
    print("\n[OK] Bearing estimation tests complete!")


if __name__ == "__main__":
    test_bearing_estimation()