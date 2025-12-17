"""
Main synthesizer engine for HumToHarmony.

Combines oscillators, envelopes, and filters to generate audio from notes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAMPLE_RATE, DEFAULT_TEMPO
from utils.midi_utils import midi_to_freq, beats_to_seconds
from .oscillators import Oscillator, sine_wave, sawtooth_wave, square_wave, triangle_wave
from .envelopes import ADSREnvelope, apply_envelope
from .filters import lowpass_filter, highpass_filter


@dataclass
class SynthParams:
    """
    Parameters for the synthesizer.
    """
    # Oscillator
    waveform: str = "sawtooth"
    detune: float = 0.0
    unison_voices: int = 1
    unison_spread: float = 10.0
    
    # Second oscillator (optional)
    osc2_enabled: bool = False
    osc2_waveform: str = "square"
    osc2_detune: float = 0.0
    osc2_mix: float = 0.5
    
    # Filter
    filter_cutoff: float = 0.8       # Normalized (0-1)
    filter_resonance: float = 0.3
    filter_envelope: float = 0.3     # Envelope modulation amount
    
    # Amplitude envelope
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.3
    
    # Filter envelope (separate from amplitude)
    filter_attack: float = 0.01
    filter_decay: float = 0.2
    filter_sustain: float = 0.5
    filter_release: float = 0.3
    
    # Effects
    distortion: float = 0.0
    noise_amount: float = 0.0
    
    # Output
    gain: float = 0.8
    pan: float = 0.0  # -1 (left) to 1 (right)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'waveform': self.waveform,
            'detune': self.detune,
            'unison_voices': self.unison_voices,
            'filter_cutoff': self.filter_cutoff,
            'filter_resonance': self.filter_resonance,
            'attack': self.attack,
            'decay': self.decay,
            'sustain': self.sustain,
            'release': self.release,
            'gain': self.gain,
        }
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'SynthParams':
        """Create params from preset name."""
        presets = {
            "piano": cls(
                waveform="triangle",
                attack=0.01, decay=0.5, sustain=0.3, release=0.3,
                filter_cutoff=0.6, filter_envelope=0.3,
            ),
            "synth_lead": cls(
                waveform="sawtooth",
                unison_voices=3, unison_spread=15,
                attack=0.01, decay=0.2, sustain=0.7, release=0.3,
                filter_cutoff=0.7, filter_resonance=0.4,
            ),
            "pad": cls(
                waveform="sawtooth",
                unison_voices=4, unison_spread=20,
                attack=0.5, decay=0.3, sustain=0.8, release=1.0,
                filter_cutoff=0.5, filter_resonance=0.2,
            ),
            "bass": cls(
                waveform="sawtooth",
                osc2_enabled=True, osc2_waveform="square", osc2_mix=0.3,
                attack=0.01, decay=0.1, sustain=0.8, release=0.2,
                filter_cutoff=0.4, filter_resonance=0.3,
            ),
            "pluck": cls(
                waveform="sawtooth",
                attack=0.001, decay=0.3, sustain=0.0, release=0.1,
                filter_cutoff=0.8, filter_envelope=0.6,
                filter_decay=0.2, filter_sustain=0.2,
            ),
            "strings": cls(
                waveform="sawtooth",
                unison_voices=4, unison_spread=10,
                attack=0.3, decay=0.2, sustain=0.7, release=0.5,
                filter_cutoff=0.5,
            ),
            "organ": cls(
                waveform="sine",
                osc2_enabled=True, osc2_waveform="sine", osc2_detune=1200,  # Octave up
                attack=0.001, decay=0.01, sustain=1.0, release=0.05,
                filter_cutoff=0.9,
            ),
            "sub_bass": cls(
                waveform="sine",
                attack=0.01, decay=0.05, sustain=0.9, release=0.2,
                filter_cutoff=0.3,
            ),
        }
        return presets.get(preset_name, cls())


class Synthesizer:
    """
    Main synthesizer class.
    
    Renders notes to audio using the specified parameters.
    """
    
    def __init__(
        self,
        params: SynthParams = None,
        sr: int = SAMPLE_RATE,
    ):
        """
        Initialize synthesizer.
        
        Args:
            params: Synth parameters (uses defaults if None)
            sr: Sample rate
        """
        self.params = params or SynthParams()
        self.sr = sr
        
        # Create oscillators
        self.osc1 = Oscillator(
            waveform=self.params.waveform,
            detune=self.params.detune,
            unison_voices=self.params.unison_voices,
            unison_spread=self.params.unison_spread,
        )
        
        if self.params.osc2_enabled:
            self.osc2 = Oscillator(
                waveform=self.params.osc2_waveform,
                detune=self.params.osc2_detune,
            )
        else:
            self.osc2 = None
        
        # Create envelopes
        self.amp_envelope = ADSREnvelope(
            attack=self.params.attack,
            decay=self.params.decay,
            sustain=self.params.sustain,
            release=self.params.release,
        )
        
        self.filter_envelope = ADSREnvelope(
            attack=self.params.filter_attack,
            decay=self.params.filter_decay,
            sustain=self.params.filter_sustain,
            release=self.params.filter_release,
        )
    
    def render_note(
        self,
        pitch: int,
        duration: float,
        velocity: int = 100,
    ) -> np.ndarray:
        """
        Render a single note to audio.
        
        Args:
            pitch: MIDI note number
            duration: Duration in seconds
            velocity: MIDI velocity (0-127)
        
        Returns:
            Audio samples
        """
        # Calculate frequency
        freq = midi_to_freq(pitch)
        
        # Add release time to duration
        total_duration = duration + self.params.release
        
        # Generate oscillator output
        audio = self.osc1.generate(freq, total_duration, self.sr)
        
        # Mix with second oscillator if enabled
        if self.osc2:
            osc2_audio = self.osc2.generate(freq, total_duration, self.sr)
            mix = self.params.osc2_mix
            audio = audio * (1 - mix) + osc2_audio * mix
        
        # Add noise if specified
        if self.params.noise_amount > 0:
            noise = np.random.uniform(-1, 1, len(audio))
            audio = audio * (1 - self.params.noise_amount) + noise * self.params.noise_amount
        
        # Apply filter with envelope modulation
        filter_env = self.filter_envelope.generate(total_duration, self.sr, duration)
        base_cutoff = self.params.filter_cutoff * (self.sr / 2)
        
        # Simple filter envelope implementation
        if self.params.filter_envelope > 0:
            # Apply time-varying filter by processing in chunks
            audio = self._apply_modulated_filter(audio, filter_env, base_cutoff)
        else:
            audio = lowpass_filter(
                audio, base_cutoff, self.sr, self.params.filter_resonance
            )
        
        # Apply amplitude envelope
        amp_env = self.amp_envelope.generate(total_duration, self.sr, duration)
        audio = apply_envelope(audio, amp_env)
        
        # Apply velocity scaling
        velocity_scale = (velocity / 127) ** 0.5  # Square root for more natural feel
        audio *= velocity_scale
        
        # Apply distortion if specified
        if self.params.distortion > 0:
            audio = self._apply_distortion(audio, self.params.distortion)
        
        # Apply gain
        audio *= self.params.gain
        
        return audio
    
    def _apply_modulated_filter(
        self,
        audio: np.ndarray,
        envelope: np.ndarray,
        base_cutoff: float,
    ) -> np.ndarray:
        """Apply filter with envelope modulation."""
        # Process in chunks for efficiency
        chunk_size = self.sr // 50  # 20ms chunks
        output = np.zeros_like(audio)
        
        for i in range(0, len(audio), chunk_size):
            end = min(i + chunk_size, len(audio))
            chunk = audio[i:end]
            
            # Get envelope value
            env_idx = min(i, len(envelope) - 1)
            env_val = envelope[env_idx]
            
            # Modulate cutoff
            max_cutoff = self.sr / 2 * 0.9
            cutoff = base_cutoff + env_val * self.params.filter_envelope * (max_cutoff - base_cutoff)
            cutoff = min(cutoff, max_cutoff)
            
            # Filter chunk
            if len(chunk) > 10:  # Need minimum samples for filter
                filtered = lowpass_filter(chunk, cutoff, self.sr, self.params.filter_resonance)
                output[i:end] = filtered
            else:
                output[i:end] = chunk
        
        return output
    
    def _apply_distortion(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Apply soft-clipping distortion."""
        # Increase gain then soft clip
        gained = audio * (1 + amount * 5)
        return np.tanh(gained) / np.tanh(1 + amount * 5)


def render_notes(
    notes: List,
    synth_params: SynthParams = None,
    tempo: float = DEFAULT_TEMPO,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Render a list of notes to audio.
    
    Args:
        notes: List of Note objects (with times in beats)
        synth_params: Synthesizer parameters
        tempo: Tempo in BPM
        sr: Sample rate
    
    Returns:
        Mixed audio output
    """
    if not notes:
        return np.zeros(int(sr * 4))  # 4 seconds of silence
    
    # Create synthesizer
    synth = Synthesizer(synth_params, sr)
    
    # Calculate total duration
    max_time = max(n.start_time + n.duration for n in notes)
    max_time_seconds = beats_to_seconds(max_time, tempo)
    
    # Add padding for release
    total_duration = max_time_seconds + synth.params.release + 0.5
    total_samples = int(total_duration * sr)
    
    # Output buffer
    output = np.zeros(total_samples)
    
    # Render each note
    for note in notes:
        # Convert beat time to seconds
        start_seconds = beats_to_seconds(note.start_time, tempo)
        duration_seconds = beats_to_seconds(note.duration, tempo)
        
        # Render note
        note_audio = synth.render_note(
            note.pitch,
            duration_seconds,
            note.velocity,
        )
        
        # Calculate sample position
        start_sample = int(start_seconds * sr)
        end_sample = start_sample + len(note_audio)
        
        # Ensure we don't go past the buffer
        if end_sample > total_samples:
            note_audio = note_audio[:total_samples - start_sample]
            end_sample = total_samples
        
        # Mix into output
        output[start_sample:end_sample] += note_audio
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 0.95:
        output = output * (0.95 / max_val)
    
    return output


def render_chord_track(
    chords: List,
    voicings: Dict[int, List[int]] = None,
    synth_params: SynthParams = None,
    tempo: float = DEFAULT_TEMPO,
    sr: int = SAMPLE_RATE,
    arpeggiate: bool = False,
) -> np.ndarray:
    """
    Render a chord progression to audio.
    
    Args:
        chords: List of Chord objects
        voicings: Optional dict mapping chord index to MIDI pitches
        synth_params: Synthesizer parameters
        tempo: Tempo in BPM
        sr: Sample rate
        arpeggiate: Arpeggiate chords instead of block chords
    
    Returns:
        Audio output
    """
    from core.pitch.detector import Note
    from core.harmony.voicing import create_chord_voicing, arpeggiate_chord, chord_to_notes
    
    all_notes = []
    
    for i, chord in enumerate(chords):
        if voicings and i in voicings:
            pitches = voicings[i]
        else:
            pitches = create_chord_voicing(chord)
        
        if arpeggiate:
            notes = arpeggiate_chord(chord, pitches, velocity=70)
        else:
            notes = [
                Note(pitch=p, start_time=chord.start_time, 
                     duration=chord.duration, velocity=70)
                for p in pitches
            ]
        
        all_notes.extend(notes)
    
    return render_notes(all_notes, synth_params, tempo, sr)

