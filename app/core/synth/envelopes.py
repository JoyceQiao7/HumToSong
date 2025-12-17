"""
ADSR envelope implementation for the synthesizer.

Envelopes shape the amplitude and filter cutoff of sounds over time.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    SAMPLE_RATE,
    DEFAULT_ATTACK,
    DEFAULT_DECAY,
    DEFAULT_SUSTAIN,
    DEFAULT_RELEASE,
)


@dataclass
class ADSREnvelope:
    """
    ADSR (Attack, Decay, Sustain, Release) envelope generator.
    
    Attributes:
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain: Sustain level (0-1)
        release: Release time in seconds
    """
    attack: float = DEFAULT_ATTACK
    decay: float = DEFAULT_DECAY
    sustain: float = DEFAULT_SUSTAIN
    release: float = DEFAULT_RELEASE
    
    def generate(
        self,
        duration: float,
        sr: int = SAMPLE_RATE,
        note_off_time: float = None,
    ) -> np.ndarray:
        """
        Generate an envelope curve.
        
        Args:
            duration: Total duration in seconds
            sr: Sample rate
            note_off_time: When note-off occurs (defaults to duration - release)
        
        Returns:
            Envelope curve as numpy array
        """
        num_samples = int(sr * duration)
        envelope = np.zeros(num_samples)
        
        # Calculate sample counts for each phase
        attack_samples = int(self.attack * sr)
        decay_samples = int(self.decay * sr)
        release_samples = int(self.release * sr)
        
        # Note-off time
        if note_off_time is None:
            note_off_time = duration - self.release
        note_off_sample = int(note_off_time * sr)
        
        current_sample = 0
        
        # Attack phase (0 to 1)
        if attack_samples > 0 and current_sample < num_samples:
            attack_end = min(attack_samples, num_samples, note_off_sample)
            envelope[current_sample:attack_end] = np.linspace(
                0, 1, attack_end - current_sample
            )
            current_sample = attack_end
        
        # Decay phase (1 to sustain)
        if decay_samples > 0 and current_sample < num_samples and current_sample < note_off_sample:
            decay_end = min(current_sample + decay_samples, num_samples, note_off_sample)
            if decay_end > current_sample:
                envelope[current_sample:decay_end] = np.linspace(
                    1, self.sustain, decay_end - current_sample
                )
            current_sample = decay_end
        
        # Sustain phase (hold at sustain level)
        if current_sample < note_off_sample and current_sample < num_samples:
            sustain_end = min(note_off_sample, num_samples)
            envelope[current_sample:sustain_end] = self.sustain
            current_sample = sustain_end
        
        # Release phase (sustain to 0)
        if current_sample < num_samples:
            release_end = min(current_sample + release_samples, num_samples)
            if release_end > current_sample:
                # Start from current level (might not be exactly sustain)
                start_level = envelope[current_sample - 1] if current_sample > 0 else self.sustain
                envelope[current_sample:release_end] = np.linspace(
                    start_level, 0, release_end - current_sample
                )
            current_sample = release_end
        
        return envelope
    
    def get_total_time(self) -> float:
        """Get minimum time needed for full envelope."""
        return self.attack + self.decay + self.release
    
    @classmethod
    def from_preset(cls, preset: str) -> 'ADSREnvelope':
        """
        Create envelope from preset name.
        
        Presets:
            - "pluck": Short attack, quick decay
            - "pad": Long attack and release
            - "organ": Instant attack, full sustain
            - "piano": Quick attack, moderate decay
            - "string": Moderate attack and release
        """
        presets = {
            "pluck": cls(attack=0.001, decay=0.3, sustain=0.0, release=0.1),
            "pad": cls(attack=0.5, decay=0.3, sustain=0.8, release=1.0),
            "organ": cls(attack=0.001, decay=0.01, sustain=1.0, release=0.05),
            "piano": cls(attack=0.01, decay=0.5, sustain=0.3, release=0.3),
            "string": cls(attack=0.2, decay=0.2, sustain=0.7, release=0.5),
            "bass": cls(attack=0.01, decay=0.1, sustain=0.8, release=0.2),
            "lead": cls(attack=0.01, decay=0.2, sustain=0.7, release=0.3),
            "soft": cls(attack=0.1, decay=0.2, sustain=0.6, release=0.4),
            "punchy": cls(attack=0.001, decay=0.15, sustain=0.4, release=0.15),
        }
        return presets.get(preset, cls())


def apply_envelope(
    audio: np.ndarray,
    envelope: np.ndarray,
) -> np.ndarray:
    """
    Apply an envelope to audio.
    
    Args:
        audio: Input audio samples
        envelope: Envelope curve (same length as audio)
    
    Returns:
        Audio with envelope applied
    """
    # Ensure same length
    min_len = min(len(audio), len(envelope))
    return audio[:min_len] * envelope[:min_len]


def generate_simple_envelope(
    duration: float,
    sr: int = SAMPLE_RATE,
    fade_in: float = 0.01,
    fade_out: float = 0.01,
) -> np.ndarray:
    """
    Generate a simple fade-in/fade-out envelope.
    
    Args:
        duration: Total duration in seconds
        sr: Sample rate
        fade_in: Fade in time in seconds
        fade_out: Fade out time in seconds
    
    Returns:
        Envelope curve
    """
    num_samples = int(sr * duration)
    envelope = np.ones(num_samples)
    
    fade_in_samples = int(fade_in * sr)
    fade_out_samples = int(fade_out * sr)
    
    if fade_in_samples > 0 and fade_in_samples < num_samples:
        envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
    
    if fade_out_samples > 0 and fade_out_samples < num_samples:
        envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
    
    return envelope


def exponential_decay(
    duration: float,
    sr: int = SAMPLE_RATE,
    decay_rate: float = 5.0,
) -> np.ndarray:
    """
    Generate an exponential decay envelope.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        decay_rate: Decay rate (higher = faster decay)
    
    Returns:
        Envelope curve
    """
    t = np.linspace(0, duration, int(sr * duration))
    return np.exp(-decay_rate * t)


def percussive_envelope(
    duration: float,
    sr: int = SAMPLE_RATE,
    attack: float = 0.001,
    decay: float = 0.3,
) -> np.ndarray:
    """
    Generate a percussive (no sustain) envelope.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        attack: Attack time
        decay: Decay time
    
    Returns:
        Envelope curve
    """
    return ADSREnvelope(
        attack=attack,
        decay=decay,
        sustain=0.0,
        release=0.0,
    ).generate(duration, sr)

