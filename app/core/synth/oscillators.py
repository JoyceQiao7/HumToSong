"""
Oscillator implementations for the synthesizer.

Provides basic waveform generators:
- Sine
- Triangle
- Sawtooth
- Square
- Noise
"""

import numpy as np
from typing import Union, Callable
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAMPLE_RATE


def sine_wave(
    frequency: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Generate a sine wave.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sr: Sample rate
        phase: Initial phase in radians
    
    Returns:
        Audio samples as numpy array
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t + phase)


def triangle_wave(
    frequency: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Generate a triangle wave.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Triangle wave using arcsin of sine
    return 2 * np.arcsin(np.sin(2 * np.pi * frequency * t + phase)) / np.pi


def sawtooth_wave(
    frequency: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Generate a sawtooth wave.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sawtooth using modulo
    return 2 * ((frequency * t + phase / (2 * np.pi)) % 1) - 1


def square_wave(
    frequency: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    phase: float = 0.0,
    duty_cycle: float = 0.5,
) -> np.ndarray:
    """
    Generate a square wave with adjustable duty cycle.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sr: Sample rate
        phase: Initial phase in radians
        duty_cycle: Duty cycle (0-1), 0.5 = standard square
    
    Returns:
        Audio samples
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Square wave using sign of sine with duty cycle
    cycle_position = (frequency * t + phase / (2 * np.pi)) % 1
    return np.where(cycle_position < duty_cycle, 1.0, -1.0)


def noise(
    duration: float,
    sr: int = SAMPLE_RATE,
    noise_type: str = "white",
) -> np.ndarray:
    """
    Generate noise.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        noise_type: "white" or "pink"
    
    Returns:
        Audio samples
    """
    num_samples = int(sr * duration)
    
    if noise_type == "white":
        return np.random.uniform(-1, 1, num_samples)
    
    elif noise_type == "pink":
        # Pink noise using Voss-McCartney algorithm (simplified)
        white = np.random.uniform(-1, 1, num_samples)
        # Apply simple lowpass filtering approximation
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        from scipy.signal import lfilter
        pink = lfilter(b, a, white)
        # Normalize
        return pink / np.max(np.abs(pink) + 1e-10)
    
    return np.random.uniform(-1, 1, num_samples)


@dataclass
class Oscillator:
    """
    Configurable oscillator with multiple waveforms and parameters.
    """
    waveform: str = "sawtooth"  # sine, triangle, sawtooth, square, noise
    detune: float = 0.0         # Detune in cents (-100 to 100)
    unison_voices: int = 1      # Number of unison voices
    unison_spread: float = 0.1  # Spread of unison voices in cents
    phase_offset: float = 0.0   # Phase offset (0-1)
    
    # Waveform generators
    _waveform_funcs: dict = field(default_factory=lambda: {
        "sine": sine_wave,
        "triangle": triangle_wave,
        "sawtooth": sawtooth_wave,
        "square": square_wave,
    })
    
    def generate(
        self,
        frequency: float,
        duration: float,
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Generate audio for this oscillator.
        
        Args:
            frequency: Base frequency in Hz
            duration: Duration in seconds
            sr: Sample rate
        
        Returns:
            Audio samples
        """
        if self.waveform == "noise":
            return noise(duration, sr)
        
        # Get waveform function
        wave_func = self._waveform_funcs.get(self.waveform, sine_wave)
        
        # Apply detune (cents to frequency ratio)
        detune_ratio = 2 ** (self.detune / 1200)
        base_freq = frequency * detune_ratio
        
        if self.unison_voices <= 1:
            # Single voice
            return wave_func(
                base_freq,
                duration,
                sr,
                phase=self.phase_offset * 2 * np.pi,
            )
        
        # Multiple unison voices
        output = np.zeros(int(sr * duration))
        
        for i in range(self.unison_voices):
            # Calculate spread for this voice
            if self.unison_voices > 1:
                spread = (i / (self.unison_voices - 1) - 0.5) * self.unison_spread
            else:
                spread = 0
            
            voice_freq = base_freq * (2 ** (spread / 1200))
            voice_phase = (self.phase_offset + i * 0.1) * 2 * np.pi
            
            voice = wave_func(voice_freq, duration, sr, phase=voice_phase)
            output += voice
        
        # Normalize by number of voices
        return output / self.unison_voices


def create_wavetable(
    waveform: str,
    table_size: int = 2048,
) -> np.ndarray:
    """
    Create a wavetable for efficient oscillator playback.
    
    Args:
        waveform: Waveform type
        table_size: Number of samples in table
    
    Returns:
        Wavetable array
    """
    t = np.linspace(0, 1, table_size, endpoint=False)
    
    if waveform == "sine":
        return np.sin(2 * np.pi * t)
    elif waveform == "triangle":
        return 2 * np.abs(2 * t - 1) - 1
    elif waveform == "sawtooth":
        return 2 * t - 1
    elif waveform == "square":
        return np.where(t < 0.5, 1.0, -1.0)
    else:
        return np.sin(2 * np.pi * t)


def blend_waveforms(
    waveforms: list,
    weights: list,
    frequency: float,
    duration: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Blend multiple waveforms together.
    
    Args:
        waveforms: List of waveform names
        weights: Mixing weights (should sum to 1)
        frequency: Frequency in Hz
        duration: Duration in seconds
        sr: Sample rate
    
    Returns:
        Blended audio
    """
    funcs = {
        "sine": sine_wave,
        "triangle": triangle_wave,
        "sawtooth": sawtooth_wave,
        "square": square_wave,
    }
    
    output = np.zeros(int(sr * duration))
    
    for waveform, weight in zip(waveforms, weights):
        if waveform in funcs:
            wave = funcs[waveform](frequency, duration, sr)
            output += wave * weight
    
    return output

