"""
Audio effects for HumToHarmony.

Provides reverb, delay, chorus, and dynamics processing.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from scipy import signal

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAMPLE_RATE


def apply_reverb(
    audio: np.ndarray,
    amount: float = 0.3,
    room_size: float = 0.5,
    damping: float = 0.5,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Apply reverb effect using a simple algorithmic reverb.
    
    Args:
        audio: Input audio
        amount: Wet/dry mix (0-1)
        room_size: Room size (0-1)
        damping: High frequency damping (0-1)
        sr: Sample rate
    
    Returns:
        Processed audio
    """
    if amount <= 0:
        return audio
    
    # Simple reverb using multiple comb filters and all-pass filters
    wet = np.zeros_like(audio)
    
    # Comb filter delay times (in samples) - based on prime numbers for diffusion
    comb_delays = [int(sr * d) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
    comb_gains = [0.805, 0.827, 0.783, 0.764]
    
    # Apply comb filters
    for delay, gain in zip(comb_delays, comb_gains):
        delay = int(delay * room_size + 100)
        filtered = _comb_filter(audio, delay, gain * (1 - damping * 0.3))
        wet += filtered
    
    wet /= len(comb_delays)
    
    # Apply all-pass filters for diffusion
    allpass_delays = [int(sr * d) for d in [0.005, 0.0017]]
    
    for delay in allpass_delays:
        wet = _allpass_filter(wet, delay, 0.7)
    
    # Mix wet and dry
    return audio * (1 - amount) + wet * amount


def _comb_filter(audio: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    """Apply a comb filter."""
    output = np.zeros(len(audio) + delay)
    output[:len(audio)] = audio
    
    for i in range(delay, len(output)):
        output[i] += output[i - delay] * feedback
    
    return output[:len(audio)]


def _allpass_filter(audio: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """Apply an all-pass filter."""
    output = np.zeros(len(audio) + delay)
    buffer = np.zeros(delay)
    
    for i in range(len(audio)):
        buf_out = buffer[i % delay]
        buffer[i % delay] = audio[i] + buf_out * gain
        output[i] = buf_out - audio[i] * gain
    
    return output[:len(audio)]


def apply_delay(
    audio: np.ndarray,
    amount: float = 0.3,
    delay_time: float = 0.25,
    feedback: float = 0.4,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Apply delay effect.
    
    Args:
        audio: Input audio
        amount: Wet/dry mix (0-1)
        delay_time: Delay time in seconds
        feedback: Feedback amount (0-0.95)
        sr: Sample rate
    
    Returns:
        Processed audio
    """
    if amount <= 0:
        return audio
    
    delay_samples = int(delay_time * sr)
    feedback = min(0.95, max(0, feedback))  # Prevent runaway feedback
    
    # Create delay buffer
    output = np.zeros(len(audio) + delay_samples * 5)  # Extra space for feedback
    output[:len(audio)] = audio
    
    # Apply delay with feedback
    wet = np.zeros_like(output)
    
    for tap in range(1, 6):  # 5 delay taps
        tap_start = delay_samples * tap
        tap_gain = (feedback ** tap) * amount
        
        if tap_start < len(output):
            wet[tap_start:tap_start + len(audio)] += audio * tap_gain
    
    # Trim to original length
    output = output[:len(audio)]
    wet = wet[:len(audio)]
    
    return audio + wet


def apply_chorus(
    audio: np.ndarray,
    amount: float = 0.3,
    rate: float = 1.5,
    depth: float = 0.002,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Apply chorus effect.
    
    Args:
        audio: Input audio
        amount: Wet/dry mix (0-1)
        rate: LFO rate in Hz
        depth: Modulation depth in seconds
        sr: Sample rate
    
    Returns:
        Processed audio
    """
    if amount <= 0:
        return audio
    
    # Create LFO
    num_samples = len(audio)
    t = np.arange(num_samples) / sr
    
    # Two LFOs slightly out of phase
    lfo1 = np.sin(2 * np.pi * rate * t)
    lfo2 = np.sin(2 * np.pi * rate * t + np.pi * 0.5)
    
    # Convert depth to samples
    depth_samples = depth * sr
    base_delay = int(0.01 * sr)  # 10ms base delay
    
    # Create modulated delays
    wet1 = np.zeros_like(audio)
    wet2 = np.zeros_like(audio)
    
    for i in range(num_samples):
        delay1 = int(base_delay + lfo1[i] * depth_samples)
        delay2 = int(base_delay + lfo2[i] * depth_samples)
        
        if i - delay1 >= 0:
            wet1[i] = audio[i - delay1]
        if i - delay2 >= 0:
            wet2[i] = audio[i - delay2]
    
    # Mix
    wet = (wet1 + wet2) / 2
    return audio * (1 - amount) + wet * amount


def apply_compression(
    audio: np.ndarray,
    threshold: float = -20,
    ratio: float = 4,
    attack: float = 0.01,
    release: float = 0.1,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Apply dynamic compression.
    
    Args:
        audio: Input audio
        threshold: Threshold in dB
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
        sr: Sample rate
    
    Returns:
        Compressed audio
    """
    # Convert threshold to linear
    threshold_linear = 10 ** (threshold / 20)
    
    # Calculate envelope
    attack_samples = int(attack * sr)
    release_samples = int(release * sr)
    
    envelope = np.zeros(len(audio))
    env_val = 0
    
    for i, sample in enumerate(np.abs(audio)):
        if sample > env_val:
            coef = 1 - np.exp(-1 / max(1, attack_samples))
        else:
            coef = 1 - np.exp(-1 / max(1, release_samples))
        
        env_val += coef * (sample - env_val)
        envelope[i] = env_val
    
    # Calculate gain reduction
    gain = np.ones_like(envelope)
    above_threshold = envelope > threshold_linear
    
    gain[above_threshold] = threshold_linear + (envelope[above_threshold] - threshold_linear) / ratio
    gain[above_threshold] /= envelope[above_threshold]
    
    # Apply gain
    return audio * gain


def apply_limiter(
    audio: np.ndarray,
    ceiling: float = 0.95,
    release: float = 0.01,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Apply a brick-wall limiter.
    
    Args:
        audio: Input audio
        ceiling: Maximum output level
        release: Release time in seconds
        sr: Sample rate
    
    Returns:
        Limited audio
    """
    # Simple limiting with soft knee
    output = audio.copy()
    
    # Find peaks
    peaks = np.abs(output)
    
    # Calculate gain reduction needed
    gain = np.ones_like(peaks)
    above_ceiling = peaks > ceiling
    gain[above_ceiling] = ceiling / peaks[above_ceiling]
    
    # Smooth gain changes
    release_samples = int(release * sr)
    if release_samples > 0:
        # Simple smoothing
        from scipy.ndimage import uniform_filter1d
        gain = uniform_filter1d(gain, release_samples)
    
    return output * gain


@dataclass
class EffectsChain:
    """
    Chain of audio effects to apply.
    """
    reverb: float = 0.0
    reverb_size: float = 0.5
    delay: float = 0.0
    delay_time: float = 0.25
    delay_feedback: float = 0.4
    chorus: float = 0.0
    compression: bool = False
    limiter: bool = True
    
    def process(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Apply all effects in the chain.
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            Processed audio
        """
        output = audio.copy()
        
        # Apply effects in order
        if self.chorus > 0:
            output = apply_chorus(output, self.chorus, sr=sr)
        
        if self.delay > 0:
            output = apply_delay(
                output, 
                self.delay, 
                self.delay_time, 
                self.delay_feedback,
                sr=sr
            )
        
        if self.reverb > 0:
            output = apply_reverb(output, self.reverb, self.reverb_size, sr=sr)
        
        if self.compression:
            output = apply_compression(output, sr=sr)
        
        if self.limiter:
            output = apply_limiter(output, sr=sr)
        
        return output

