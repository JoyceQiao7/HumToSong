"""
Audio filters for the synthesizer.

Provides lowpass, highpass, and bandpass filters.
"""

import numpy as np
from scipy import signal
from typing import Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAMPLE_RATE


def lowpass_filter(
    audio: np.ndarray,
    cutoff: float,
    sr: int = SAMPLE_RATE,
    resonance: float = 0.707,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a lowpass filter.
    
    Args:
        audio: Input audio
        cutoff: Cutoff frequency in Hz (or normalized 0-1 if < 1)
        sr: Sample rate
        resonance: Filter resonance (Q factor), 0.707 = Butterworth
        order: Filter order
    
    Returns:
        Filtered audio
    """
    # Convert normalized cutoff to Hz if needed
    if cutoff < 1:
        cutoff = cutoff * (sr / 2)
    
    # Clamp cutoff to valid range
    nyquist = sr / 2
    cutoff = min(cutoff, nyquist * 0.99)
    cutoff = max(cutoff, 20)
    
    # Normalize cutoff
    normalized_cutoff = cutoff / nyquist
    
    # Design filter
    if resonance != 0.707:
        # Use biquad for resonance control
        b, a = _biquad_lpf(normalized_cutoff, resonance)
    else:
        # Butterworth for flat response
        b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    # Apply filter (use filtfilt for zero phase)
    try:
        return signal.filtfilt(b, a, audio)
    except ValueError:
        # If audio is too short, use lfilter
        return signal.lfilter(b, a, audio)


def highpass_filter(
    audio: np.ndarray,
    cutoff: float,
    sr: int = SAMPLE_RATE,
    resonance: float = 0.707,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a highpass filter.
    
    Args:
        audio: Input audio
        cutoff: Cutoff frequency in Hz (or normalized 0-1)
        sr: Sample rate
        resonance: Filter resonance
        order: Filter order
    
    Returns:
        Filtered audio
    """
    # Convert normalized cutoff to Hz if needed
    if cutoff < 1:
        cutoff = cutoff * (sr / 2)
    
    # Clamp cutoff
    nyquist = sr / 2
    cutoff = min(cutoff, nyquist * 0.99)
    cutoff = max(cutoff, 20)
    
    normalized_cutoff = cutoff / nyquist
    
    if resonance != 0.707:
        b, a = _biquad_hpf(normalized_cutoff, resonance)
    else:
        b, a = signal.butter(order, normalized_cutoff, btype='high')
    
    try:
        return signal.filtfilt(b, a, audio)
    except ValueError:
        return signal.lfilter(b, a, audio)


def bandpass_filter(
    audio: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sr: int = SAMPLE_RATE,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a bandpass filter.
    
    Args:
        audio: Input audio
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        sr: Sample rate
        order: Filter order
    
    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    
    low = max(low_cutoff, 20) / nyquist
    high = min(high_cutoff, nyquist * 0.99) / nyquist
    
    if low >= high:
        return audio
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    try:
        return signal.filtfilt(b, a, audio)
    except ValueError:
        return signal.lfilter(b, a, audio)


def _biquad_lpf(cutoff: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a biquad lowpass filter with resonance.
    
    Args:
        cutoff: Normalized cutoff (0-1)
        q: Q factor (resonance)
    
    Returns:
        Tuple of (b, a) filter coefficients
    """
    w0 = np.pi * cutoff
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])
    
    return b, a


def _biquad_hpf(cutoff: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a biquad highpass filter with resonance.
    """
    w0 = np.pi * cutoff
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])
    
    return b, a


class FilterEnvelope:
    """
    Filter with envelope modulation.
    """
    
    def __init__(
        self,
        filter_type: str = "lowpass",
        base_cutoff: float = 1000,
        envelope_amount: float = 0.5,
        resonance: float = 0.707,
    ):
        """
        Args:
            filter_type: "lowpass", "highpass", or "bandpass"
            base_cutoff: Base cutoff frequency in Hz
            envelope_amount: How much envelope affects cutoff (0-1)
            resonance: Filter resonance
        """
        self.filter_type = filter_type
        self.base_cutoff = base_cutoff
        self.envelope_amount = envelope_amount
        self.resonance = resonance
    
    def apply(
        self,
        audio: np.ndarray,
        envelope: np.ndarray,
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Apply filter with envelope modulation.
        
        For efficiency, we divide the audio into chunks and apply
        different filter settings to each chunk.
        """
        # Number of chunks
        chunk_size = sr // 20  # 50ms chunks
        num_chunks = len(audio) // chunk_size + 1
        
        output = np.zeros_like(audio)
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(audio))
            
            if start >= len(audio):
                break
            
            # Get envelope value for this chunk
            env_idx = min(start, len(envelope) - 1)
            env_value = envelope[env_idx]
            
            # Calculate modulated cutoff
            max_cutoff = sr / 2 * 0.9
            cutoff_range = max_cutoff - self.base_cutoff
            cutoff = self.base_cutoff + env_value * self.envelope_amount * cutoff_range
            
            # Apply filter to chunk
            chunk = audio[start:end]
            
            if self.filter_type == "lowpass":
                filtered = lowpass_filter(chunk, cutoff, sr, self.resonance)
            elif self.filter_type == "highpass":
                filtered = highpass_filter(chunk, cutoff, sr, self.resonance)
            else:
                filtered = chunk
            
            output[start:end] = filtered
        
        return output


def apply_eq(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    low_gain: float = 0.0,
    mid_gain: float = 0.0,
    high_gain: float = 0.0,
) -> np.ndarray:
    """
    Simple 3-band EQ.
    
    Args:
        audio: Input audio
        sr: Sample rate
        low_gain: Low frequency gain in dB (-12 to +12)
        mid_gain: Mid frequency gain in dB
        high_gain: High frequency gain in dB
    
    Returns:
        EQ'd audio
    """
    # Split into bands
    low = lowpass_filter(audio, 200, sr)
    mid = bandpass_filter(audio, 200, 4000, sr)
    high = highpass_filter(audio, 4000, sr)
    
    # Apply gains (convert dB to linear)
    low *= 10 ** (low_gain / 20)
    mid *= 10 ** (mid_gain / 20)
    high *= 10 ** (high_gain / 20)
    
    # Combine
    return low + mid + high

