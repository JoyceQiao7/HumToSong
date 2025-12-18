"""
MIDI utilities for HumToHarmony.

Provides functions for converting between:
- Note names and MIDI numbers
- Frequencies and MIDI numbers
- Note durations and beats
"""

import numpy as np
from typing import Tuple, Optional

# Note name constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ENHARMONIC_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# MIDI reference: A4 = MIDI 69 = 440 Hz
A4_MIDI = 69
A4_FREQ = 440.0


def note_to_midi(note_name: str) -> int:
    """
    Convert note name to MIDI number.
    
    Args:
        note_name: Note name like 'C4', 'F#3', 'Bb5'
    
    Returns:
        MIDI note number (0-127)
    
    Examples:
        >>> note_to_midi('C4')
        60
        >>> note_to_midi('A4')
        69
        >>> note_to_midi('F#3')
        54
    """
    note_name = note_name.strip()
    
    # Parse the note name
    if len(note_name) < 2:
        raise ValueError(f"Invalid note name: {note_name}")
    
    # Handle sharps and flats
    if note_name[1] in ['#', 'b']:
        pitch_class = note_name[:2]
        octave = int(note_name[2:])
    else:
        pitch_class = note_name[0]
        octave = int(note_name[1:])
    
    # Convert pitch class to semitone offset
    pitch_class_upper = pitch_class.upper()
    
    if pitch_class_upper in NOTE_NAMES:
        semitone = NOTE_NAMES.index(pitch_class_upper)
    elif pitch_class_upper in ENHARMONIC_NAMES:
        semitone = ENHARMONIC_NAMES.index(pitch_class_upper)
    else:
        raise ValueError(f"Unknown pitch class: {pitch_class}")
    
    # Calculate MIDI number (C4 = 60)
    midi_number = (octave + 1) * 12 + semitone
    
    return midi_number


def midi_to_note(midi_number: int, use_flats: bool = False) -> str:
    """
    Convert MIDI number to note name.
    
    Args:
        midi_number: MIDI note number (0-127)
        use_flats: Use flat names instead of sharp names
    
    Returns:
        Note name string like 'C4', 'F#3'
    
    Examples:
        >>> midi_to_note(60)
        'C4'
        >>> midi_to_note(69)
        'A4'
        >>> midi_to_note(70, use_flats=True)
        'Bb4'
    """
    if not 0 <= midi_number <= 127:
        raise ValueError(f"MIDI number must be 0-127, got {midi_number}")
    
    octave = (midi_number // 12) - 1
    semitone = midi_number % 12
    
    names = ENHARMONIC_NAMES if use_flats else NOTE_NAMES
    note_name = names[semitone]
    
    return f"{note_name}{octave}"


def midi_to_freq(midi_number: float) -> float:
    """
    Convert MIDI number to frequency in Hz.
    
    Supports fractional MIDI numbers for microtonal pitches.
    
    Args:
        midi_number: MIDI note number (can be fractional)
    
    Returns:
        Frequency in Hz
    
    Examples:
        >>> midi_to_freq(69)  # A4
        440.0
        >>> midi_to_freq(60)  # C4
        261.625...
    """
    return A4_FREQ * (2 ** ((midi_number - A4_MIDI) / 12))


def freq_to_midi(frequency: float) -> float:
    """
    Convert frequency to MIDI number.
    
    Returns fractional MIDI number for frequencies between semitones.
    
    Args:
        frequency: Frequency in Hz
    
    Returns:
        MIDI note number (fractional), or array of MIDI numbers
    
    Examples:
        >>> freq_to_midi(440.0)
        69.0
        >>> freq_to_midi(261.63)
        60.0...
    """
    # Handle both scalar and array inputs
    frequency = np.asarray(frequency)
    
    # Check for invalid frequencies
    if np.any(frequency <= 0):
        # For arrays, just warn but continue (caller should have filtered)
        if frequency.size > 1:
            pass  # Caller should have already filtered out zeros
        else:
            raise ValueError(f"Frequency must be positive, got {frequency}")
    
    return A4_MIDI + 12 * np.log2(frequency / A4_FREQ)


def quantize_pitch(midi_number: float, round_mode: str = 'nearest') -> int:
    """
    Quantize a fractional MIDI number to the nearest semitone.
    
    Args:
        midi_number: Fractional MIDI number
        round_mode: 'nearest', 'up', or 'down'
    
    Returns:
        Integer MIDI number
    """
    if round_mode == 'nearest':
        return int(round(midi_number))
    elif round_mode == 'up':
        return int(np.ceil(midi_number))
    elif round_mode == 'down':
        return int(np.floor(midi_number))
    else:
        raise ValueError(f"Unknown round mode: {round_mode}")


def get_scale_pitches(root: str, scale_type: str = 'major') -> list:
    """
    Get all MIDI pitches in a scale for all octaves (0-127).
    
    Args:
        root: Root note name (e.g., 'C', 'F#')
        scale_type: Scale type ('major', 'minor', 'harmonic_minor', etc.)
    
    Returns:
        List of valid MIDI pitches in the scale
    """
    # Scale intervals in semitones
    SCALE_INTERVALS = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],  # Natural minor
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'locrian': [0, 1, 3, 5, 6, 8, 10],
        'pentatonic_major': [0, 2, 4, 7, 9],
        'pentatonic_minor': [0, 3, 5, 7, 10],
        'blues': [0, 3, 5, 6, 7, 10],
        'chromatic': list(range(12)),
    }
    
    if scale_type not in SCALE_INTERVALS:
        raise ValueError(f"Unknown scale type: {scale_type}")
    
    # Get root pitch class
    root_upper = root.upper()
    if root_upper in NOTE_NAMES:
        root_pc = NOTE_NAMES.index(root_upper)
    elif root_upper in ENHARMONIC_NAMES:
        root_pc = ENHARMONIC_NAMES.index(root_upper)
    else:
        raise ValueError(f"Unknown root note: {root}")
    
    # Generate all pitches in the scale
    intervals = SCALE_INTERVALS[scale_type]
    pitches = []
    
    for octave in range(-1, 11):  # MIDI octaves -1 to 10
        for interval in intervals:
            pitch = (octave + 1) * 12 + root_pc + interval
            if 0 <= pitch <= 127:
                pitches.append(pitch)
    
    return sorted(set(pitches))


def snap_to_scale(midi_number: float, scale_pitches: list) -> int:
    """
    Snap a MIDI pitch to the nearest pitch in a scale.
    
    Args:
        midi_number: Input MIDI pitch (can be fractional)
        scale_pitches: List of valid MIDI pitches in the scale
    
    Returns:
        Nearest valid MIDI pitch in the scale
    """
    if not scale_pitches:
        return int(round(midi_number))
    
    # Find nearest scale pitch
    diffs = [abs(p - midi_number) for p in scale_pitches]
    nearest_idx = np.argmin(diffs)
    
    return scale_pitches[nearest_idx]


def beats_to_seconds(beats: float, tempo: float) -> float:
    """
    Convert beats to seconds.
    
    Args:
        beats: Number of beats
        tempo: Tempo in BPM
    
    Returns:
        Duration in seconds
    """
    return beats * 60.0 / tempo


def seconds_to_beats(seconds: float, tempo: float) -> float:
    """
    Convert seconds to beats.
    
    Args:
        seconds: Duration in seconds
        tempo: Tempo in BPM
    
    Returns:
        Number of beats
    """
    return seconds * tempo / 60.0


def quantize_beat(beat: float, resolution: int = 16) -> float:
    """
    Quantize a beat position to a grid.
    
    Args:
        beat: Beat position
        resolution: Grid resolution (16 = 16th notes, 8 = 8th notes, etc.)
    
    Returns:
        Quantized beat position
    """
    grid = 4.0 / resolution  # Grid size in beats
    return round(beat / grid) * grid


def get_chord_midi_notes(root: str, chord_type: str = 'major', octave: int = 4) -> list:
    """
    Get MIDI notes for a chord.
    
    Args:
        root: Root note (e.g., 'C', 'F#')
        chord_type: Chord type ('major', 'minor', '7', 'maj7', etc.)
        octave: Octave for root note
    
    Returns:
        List of MIDI note numbers
    """
    # Chord intervals in semitones from root
    CHORD_INTERVALS = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'dim': [0, 3, 6],
        'aug': [0, 4, 8],
        '7': [0, 4, 7, 10],
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dim7': [0, 3, 6, 9],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'add9': [0, 4, 7, 14],
        '6': [0, 4, 7, 9],
        'min6': [0, 3, 7, 9],
    }
    
    if chord_type not in CHORD_INTERVALS:
        raise ValueError(f"Unknown chord type: {chord_type}")
    
    # Get root MIDI number
    root_midi = note_to_midi(f"{root}{octave}")
    
    # Build chord
    return [root_midi + interval for interval in CHORD_INTERVALS[chord_type]]

