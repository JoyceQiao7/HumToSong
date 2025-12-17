"""
Track management and mixing for HumToHarmony.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAMPLE_RATE
from .effects import EffectsChain


@dataclass
class Track:
    """
    Represents an audio track (melody, harmony, or bass).
    """
    name: str
    audio: np.ndarray = field(default_factory=lambda: np.array([]))
    volume: float = 0.8       # 0-1
    pan: float = 0.0          # -1 (left) to 1 (right)
    mute: bool = False
    solo: bool = False
    effects: EffectsChain = field(default_factory=EffectsChain)
    
    def get_processed_audio(self, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Get audio with effects applied."""
        if len(self.audio) == 0 or self.mute:
            return np.array([])
        
        # Apply effects
        processed = self.effects.process(self.audio, sr)
        
        # Apply volume
        processed *= self.volume
        
        return processed
    
    def get_stereo_audio(self, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Get audio as stereo with pan applied."""
        mono = self.get_processed_audio(sr)
        
        if len(mono) == 0:
            return np.array([[], []])
        
        # Calculate pan gains
        # Using constant-power panning
        angle = (self.pan + 1) * np.pi / 4  # 0 to pi/2
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)
        
        # Create stereo
        left = mono * left_gain
        right = mono * right_gain
        
        return np.array([left, right])


def mix_tracks(
    tracks: List[Track],
    sr: int = SAMPLE_RATE,
    normalize: bool = True,
    master_effects: EffectsChain = None,
) -> np.ndarray:
    """
    Mix multiple tracks together.
    
    Args:
        tracks: List of Track objects
        sr: Sample rate
        normalize: Normalize output to prevent clipping
        master_effects: Effects to apply to master bus
    
    Returns:
        Mixed stereo audio (shape: 2 x N)
    """
    if not tracks:
        return np.zeros((2, int(sr * 4)))  # 4 seconds of silence
    
    # Check for solo tracks
    solo_tracks = [t for t in tracks if t.solo]
    active_tracks = solo_tracks if solo_tracks else tracks
    
    # Find maximum length
    max_length = max(
        len(t.audio) for t in active_tracks if len(t.audio) > 0
    ) if active_tracks else int(sr * 4)
    
    # Mix into stereo bus
    mix = np.zeros((2, max_length))
    
    for track in active_tracks:
        if track.mute or len(track.audio) == 0:
            continue
        
        stereo = track.get_stereo_audio(sr)
        
        if stereo.shape[1] > 0:
            # Add to mix (pad if necessary)
            length = min(stereo.shape[1], max_length)
            mix[0, :length] += stereo[0, :length]
            mix[1, :length] += stereo[1, :length]
    
    # Apply master effects
    if master_effects:
        mix[0] = master_effects.process(mix[0], sr)
        mix[1] = master_effects.process(mix[1], sr)
    
    # Normalize
    if normalize:
        max_val = np.max(np.abs(mix))
        if max_val > 0.95:
            mix = mix * (0.95 / max_val)
    
    return mix


def mix_to_mono(stereo: np.ndarray) -> np.ndarray:
    """Convert stereo mix to mono."""
    if stereo.ndim == 1:
        return stereo
    return (stereo[0] + stereo[1]) / 2


def create_track_from_notes(
    notes: List,
    synth_params,
    track_name: str,
    tempo: float,
    sr: int = SAMPLE_RATE,
    effects: EffectsChain = None,
) -> Track:
    """
    Create a Track from a list of notes.
    
    Args:
        notes: List of Note objects
        synth_params: SynthParams for rendering
        track_name: Name for the track
        tempo: Tempo in BPM
        sr: Sample rate
        effects: Effects chain for the track
    
    Returns:
        Track object with rendered audio
    """
    from core.synth.engine import render_notes
    
    audio = render_notes(notes, synth_params, tempo, sr)
    
    return Track(
        name=track_name,
        audio=audio,
        effects=effects or EffectsChain(),
    )


class Mixer:
    """
    Complete mixer with track management.
    """
    
    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.tracks: Dict[str, Track] = {}
        self.master_volume = 0.9
        self.master_effects = EffectsChain(limiter=True)
    
    def add_track(self, name: str, audio: np.ndarray, **kwargs) -> Track:
        """Add a track to the mixer."""
        track = Track(name=name, audio=audio, **kwargs)
        self.tracks[name] = track
        return track
    
    def get_track(self, name: str) -> Optional[Track]:
        """Get a track by name."""
        return self.tracks.get(name)
    
    def remove_track(self, name: str):
        """Remove a track."""
        if name in self.tracks:
            del self.tracks[name]
    
    def set_volume(self, track_name: str, volume: float):
        """Set track volume."""
        if track_name in self.tracks:
            self.tracks[track_name].volume = max(0, min(1, volume))
    
    def set_pan(self, track_name: str, pan: float):
        """Set track pan."""
        if track_name in self.tracks:
            self.tracks[track_name].pan = max(-1, min(1, pan))
    
    def mute_track(self, track_name: str, mute: bool = True):
        """Mute/unmute a track."""
        if track_name in self.tracks:
            self.tracks[track_name].mute = mute
    
    def solo_track(self, track_name: str, solo: bool = True):
        """Solo/unsolo a track."""
        if track_name in self.tracks:
            self.tracks[track_name].solo = solo
    
    def mix(self, normalize: bool = True) -> np.ndarray:
        """Mix all tracks and return stereo output."""
        mix = mix_tracks(
            list(self.tracks.values()),
            self.sr,
            normalize=False,
            master_effects=None,
        )
        
        # Apply master volume
        mix *= self.master_volume
        
        # Apply master effects
        mix[0] = self.master_effects.process(mix[0], self.sr)
        mix[1] = self.master_effects.process(mix[1], self.sr)
        
        # Final normalize
        if normalize:
            max_val = np.max(np.abs(mix))
            if max_val > 0.95:
                mix = mix * (0.95 / max_val)
        
        return mix
    
    def get_mono_mix(self) -> np.ndarray:
        """Get mono mixdown."""
        stereo = self.mix()
        return mix_to_mono(stereo)

