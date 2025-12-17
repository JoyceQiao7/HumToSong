"""
Bass line generator for HumToHarmony.

Generates bass lines that complement chord progressions.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.midi_utils import note_to_midi, NOTE_NAMES
from core.pitch.detector import Note


# Bass pattern templates for different styles
BASS_PATTERNS = {
    "pop": [
        # Quarter notes on root
        [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        # Root and fifth
        [(0, 1.0), (2, 1.0), (0, 1.0), (2, 1.0)],
        # Root, fifth, octave pattern
        [(0, 2.0), (2, 1.0), (3, 1.0)],
    ],
    "jazz": [
        # Walking bass (simplified)
        [(0, 1.0), (0.5, 0.5), (1, 0.5), (1.5, 0.5), (2, 1.0), (3, 1.0)],
        # Two-feel
        [(0, 2.0), (2, 2.0)],
    ],
    "lofi": [
        # Sparse
        [(0, 2.0), (2.5, 1.5)],
        # Very sparse
        [(0, 4.0)],
        # Syncopated
        [(0, 1.5), (1.5, 1.0), (2.5, 1.5)],
    ],
    "classical": [
        # Alberti-style (simplified)
        [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        # Root only on downbeat
        [(0, 4.0)],
    ],
    "electronic": [
        # Eighth notes
        [(0, 0.5), (0.5, 0.5), (1, 0.5), (1.5, 0.5), (2, 0.5), (2.5, 0.5), (3, 0.5), (3.5, 0.5)],
        # Offbeat
        [(0.5, 1.0), (1.5, 1.0), (2.5, 1.0), (3.5, 0.5)],
    ],
}


def generate_bass_line(
    chords: List,
    style: str = "pop",
    octave: int = 2,
    velocity: int = 90,
) -> List[Note]:
    """
    Generate a bass line from chord progression.
    
    Args:
        chords: List of Chord objects
        style: Musical style
        octave: Bass octave (usually 2 or 3)
        velocity: MIDI velocity for bass notes
    
    Returns:
        List of Note objects for bass line
    """
    if not chords:
        return []
    
    bass_notes = []
    patterns = BASS_PATTERNS.get(style, BASS_PATTERNS["pop"])
    
    for chord in chords:
        # Get root note MIDI pitch
        root_midi = note_to_midi(f"{chord.root}{octave}")
        fifth_midi = root_midi + 7  # Perfect fifth
        
        # Choose a pattern
        pattern = random.choice(patterns)
        
        # Scale pattern to chord duration
        pattern_duration = max(p[0] + p[1] for p in pattern)
        scale_factor = chord.duration / max(pattern_duration, 4.0)
        
        for beat_offset, note_duration in pattern:
            # Scale timing to chord duration
            scaled_offset = beat_offset * scale_factor
            scaled_duration = note_duration * scale_factor
            
            # Determine pitch (root or fifth)
            if beat_offset == 0 or random.random() < 0.7:
                pitch = root_midi
            else:
                pitch = fifth_midi if random.random() < 0.5 else root_midi
            
            # Create note
            note = Note(
                pitch=pitch,
                start_time=chord.start_time + scaled_offset,
                duration=scaled_duration,
                velocity=velocity,
                confidence=1.0,
            )
            bass_notes.append(note)
    
    return bass_notes


def generate_walking_bass(
    chords: List,
    octave: int = 2,
    velocity: int = 85,
) -> List[Note]:
    """
    Generate a jazz-style walking bass line.
    
    Args:
        chords: List of Chord objects
        octave: Bass octave
        velocity: MIDI velocity
    
    Returns:
        List of Note objects
    """
    if not chords:
        return []
    
    bass_notes = []
    prev_pitch = None
    
    for i, chord in enumerate(chords):
        root_midi = note_to_midi(f"{chord.root}{octave}")
        third_midi = root_midi + (3 if 'min' in chord.quality else 4)
        fifth_midi = root_midi + 7
        seventh_midi = root_midi + (10 if '7' in chord.quality else 11)
        
        # Get next chord root for approach notes
        if i < len(chords) - 1:
            next_root = note_to_midi(f"{chords[i+1].root}{octave}")
        else:
            next_root = root_midi
        
        # Walking bass: one note per beat
        beats_in_chord = int(chord.duration)
        
        for beat in range(beats_in_chord):
            if beat == 0:
                # Always play root on beat 1
                pitch = root_midi
            elif beat == beats_in_chord - 1:
                # Approach note to next chord
                pitch = _approach_note(root_midi, next_root)
            else:
                # Chord tones and passing tones
                options = [root_midi, third_midi, fifth_midi]
                if '7' in chord.quality:
                    options.append(seventh_midi)
                pitch = random.choice(options)
            
            # Add chromatic approach occasionally
            if beat == beats_in_chord - 1 and random.random() < 0.3:
                # Half-step approach
                if next_root > pitch:
                    pitch = next_root - 1
                else:
                    pitch = next_root + 1
            
            note = Note(
                pitch=pitch,
                start_time=chord.start_time + beat,
                duration=1.0,
                velocity=velocity + random.randint(-5, 5),
                confidence=1.0,
            )
            bass_notes.append(note)
            prev_pitch = pitch
    
    return bass_notes


def _approach_note(current: int, target: int) -> int:
    """Get a note that approaches the target chromatically or by step."""
    diff = target - current
    
    if abs(diff) <= 2:
        # Already close, use chromatic approach
        return target - 1 if diff > 0 else target + 1
    elif diff > 0:
        # Moving up, approach from below
        return target - random.choice([1, 2])
    else:
        # Moving down, approach from above
        return target + random.choice([1, 2])


def generate_arpeggiated_bass(
    chords: List,
    octave: int = 2,
    pattern: str = "up",
    velocity: int = 80,
) -> List[Note]:
    """
    Generate an arpeggiated bass line.
    
    Args:
        chords: List of Chord objects
        octave: Bass octave
        pattern: "up", "down", or "updown"
        velocity: MIDI velocity
    
    Returns:
        List of Note objects
    """
    if not chords:
        return []
    
    bass_notes = []
    
    for chord in chords:
        root_midi = note_to_midi(f"{chord.root}{octave}")
        
        # Build arpeggio pitches
        if 'min' in chord.quality:
            third = root_midi + 3
        else:
            third = root_midi + 4
        fifth = root_midi + 7
        
        if '7' in chord.quality:
            pitches = [root_midi, third, fifth, root_midi + 10]
        else:
            pitches = [root_midi, third, fifth, root_midi + 12]
        
        # Apply pattern
        if pattern == "down":
            pitches = pitches[::-1]
        elif pattern == "updown":
            pitches = pitches + pitches[-2:0:-1]
        
        # Distribute across chord duration
        note_duration = chord.duration / len(pitches)
        
        for j, pitch in enumerate(pitches):
            note = Note(
                pitch=pitch,
                start_time=chord.start_time + j * note_duration,
                duration=note_duration,
                velocity=velocity,
                confidence=1.0,
            )
            bass_notes.append(note)
    
    return bass_notes

