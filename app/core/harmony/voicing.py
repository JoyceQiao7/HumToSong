"""
Chord voicing utilities for HumToHarmony.

Creates natural-sounding chord voicings for different instruments and styles.
"""

import numpy as np
from typing import List, Optional, Tuple
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.midi_utils import note_to_midi, NOTE_NAMES
from core.pitch.detector import Note


# Chord intervals by quality (in semitones from root)
CHORD_INTERVALS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 14],
    "6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
    "9": [0, 4, 7, 10, 14],
    "min9": [0, 3, 7, 10, 14],
}


def voice_chord(
    chord,
    octave: int = 4,
    voicing_type: str = "close",
    inversion: int = 0,
) -> List[int]:
    """
    Create a voicing for a chord.
    
    Args:
        chord: Chord object with root and quality
        octave: Base octave for voicing
        voicing_type: "close", "open", "spread", or "drop2"
        inversion: 0 = root position, 1 = first inversion, etc.
    
    Returns:
        List of MIDI note numbers
    """
    # Get chord intervals
    intervals = CHORD_INTERVALS.get(chord.quality, CHORD_INTERVALS["major"])
    
    # Get root MIDI note
    root_midi = note_to_midi(f"{chord.root}{octave}")
    
    # Build basic chord
    pitches = [root_midi + interval for interval in intervals]
    
    # Apply inversion
    if inversion > 0:
        for _ in range(min(inversion, len(pitches) - 1)):
            pitches[0] += 12
            pitches = pitches[1:] + [pitches[0]]
    
    # Apply voicing type
    if voicing_type == "open":
        pitches = _open_voicing(pitches)
    elif voicing_type == "spread":
        pitches = _spread_voicing(pitches)
    elif voicing_type == "drop2":
        pitches = _drop2_voicing(pitches)
    
    return sorted(pitches)


def _open_voicing(pitches: List[int]) -> List[int]:
    """Convert close voicing to open voicing."""
    if len(pitches) < 3:
        return pitches
    
    # Move middle note(s) up an octave
    result = pitches.copy()
    for i in range(1, len(result) - 1):
        result[i] += 12
    
    return result


def _spread_voicing(pitches: List[int]) -> List[int]:
    """Spread voicing across wider range."""
    if len(pitches) < 2:
        return pitches
    
    result = [pitches[0]]
    for i, pitch in enumerate(pitches[1:], 1):
        # Alternate adding octaves
        offset = (i // 2) * 12
        result.append(pitch + offset)
    
    return result


def _drop2_voicing(pitches: List[int]) -> List[int]:
    """Drop the second-highest note down an octave."""
    if len(pitches) < 4:
        return pitches
    
    result = pitches.copy()
    # Sort and drop second from top
    result = sorted(result)
    if len(result) >= 2:
        result[-2] -= 12
    
    return sorted(result)


def create_chord_voicing(
    chord,
    style: str = "pop",
    prev_voicing: List[int] = None,
    octave: int = 4,
) -> List[int]:
    """
    Create a chord voicing appropriate for the style, with voice leading.
    
    Args:
        chord: Chord object
        style: Musical style
        prev_voicing: Previous chord's voicing for voice leading
        octave: Base octave
    
    Returns:
        List of MIDI note numbers
    """
    # Style-specific voicing preferences
    style_prefs = {
        "pop": {"type": "close", "octave_adjust": 0},
        "jazz": {"type": "drop2", "octave_adjust": 0},
        "lofi": {"type": "spread", "octave_adjust": 0},
        "classical": {"type": "close", "octave_adjust": 0},
        "electronic": {"type": "open", "octave_adjust": 1},
    }
    
    prefs = style_prefs.get(style, style_prefs["pop"])
    
    # Get base voicing
    voicing = voice_chord(
        chord,
        octave=octave + prefs["octave_adjust"],
        voicing_type=prefs["type"],
    )
    
    # Apply voice leading if we have a previous chord
    if prev_voicing:
        voicing = _apply_voice_leading(voicing, prev_voicing)
    
    return voicing


def _apply_voice_leading(
    current: List[int],
    previous: List[int],
    max_movement: int = 4,
) -> List[int]:
    """
    Adjust voicing for smooth voice leading.
    
    Tries to minimize the movement of each voice.
    """
    if not previous:
        return current
    
    result = current.copy()
    
    # Try to match number of voices
    while len(result) < len(previous):
        # Double the root
        result.append(result[0] + 12)
    
    # For each note in current chord, find closest previous note
    for i, note in enumerate(result):
        # Find closest note in previous chord
        distances = [abs(note - prev) for prev in previous]
        min_dist = min(distances)
        
        # If too far, try octave adjustment
        if min_dist > max_movement:
            # Try moving up or down an octave
            if note - 12 >= 36:  # Don't go too low
                dist_down = min(abs(note - 12 - prev) for prev in previous)
                if dist_down < min_dist:
                    result[i] = note - 12
            if note + 12 <= 96:  # Don't go too high
                dist_up = min(abs(note + 12 - prev) for prev in previous)
                if dist_up < min_dist:
                    result[i] = note + 12
    
    return sorted(result)


def chord_to_notes(
    chord,
    voicing: List[int] = None,
    velocity: int = 80,
    style: str = "pop",
) -> List[Note]:
    """
    Convert a chord to Note objects for synthesis.
    
    Args:
        chord: Chord object with timing
        voicing: Optional specific voicing (MIDI notes)
        velocity: Note velocity
        style: Style for automatic voicing
    
    Returns:
        List of Note objects
    """
    if voicing is None:
        voicing = create_chord_voicing(chord, style=style)
    
    notes = []
    for pitch in voicing:
        note = Note(
            pitch=pitch,
            start_time=chord.start_time,
            duration=chord.duration,
            velocity=velocity + random.randint(-5, 5),
            confidence=1.0,
        )
        notes.append(note)
    
    return notes


def arpeggiate_chord(
    chord,
    voicing: List[int] = None,
    pattern: str = "up",
    rate: float = 0.25,
    velocity: int = 70,
) -> List[Note]:
    """
    Create arpeggiated notes from a chord.
    
    Args:
        chord: Chord object
        voicing: MIDI notes for the chord
        pattern: "up", "down", "updown", "random"
        rate: Note duration in beats
        velocity: Note velocity
    
    Returns:
        List of Note objects
    """
    if voicing is None:
        voicing = create_chord_voicing(chord)
    
    pitches = list(voicing)
    
    # Apply pattern
    if pattern == "down":
        pitches = pitches[::-1]
    elif pattern == "updown":
        pitches = pitches + pitches[-2:0:-1]
    elif pattern == "random":
        random.shuffle(pitches)
    
    # Calculate how many notes fit in the chord duration
    num_notes = int(chord.duration / rate)
    
    notes = []
    for i in range(num_notes):
        pitch = pitches[i % len(pitches)]
        note = Note(
            pitch=pitch,
            start_time=chord.start_time + i * rate,
            duration=rate,
            velocity=velocity + random.randint(-10, 10),
            confidence=1.0,
        )
        notes.append(note)
    
    return notes

