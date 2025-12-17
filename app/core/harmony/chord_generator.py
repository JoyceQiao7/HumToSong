"""
Chord progression generator for HumToHarmony.

Generates chord progressions that fit a melody based on:
1. Melody note analysis
2. Music theory rules
3. Style templates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHORD_TEMPLATES, DEFAULT_STYLE
from utils.midi_utils import NOTE_NAMES, get_chord_midi_notes


@dataclass
class Chord:
    """Represents a chord in a progression."""
    root: str                   # Root note (e.g., "C", "F#")
    quality: str                # Chord quality (e.g., "major", "minor", "7")
    symbol: str                 # Full chord symbol (e.g., "Cmaj7", "Am")
    start_time: float          # Start time in beats
    duration: float            # Duration in beats
    
    def to_dict(self) -> dict:
        return {
            'root': self.root,
            'quality': self.quality,
            'symbol': self.symbol,
            'start_time': self.start_time,
            'duration': self.duration,
        }
    
    def get_midi_notes(self, octave: int = 4) -> List[int]:
        """Get MIDI notes for this chord."""
        return get_chord_midi_notes(self.root, self.quality, octave)


@dataclass
class ChordProgression:
    """A sequence of chords."""
    chords: List[Chord] = field(default_factory=list)
    key: str = "C"
    scale: str = "major"
    
    def __len__(self) -> int:
        return len(self.chords)
    
    def __iter__(self):
        return iter(self.chords)
    
    def __getitem__(self, idx):
        return self.chords[idx]
    
    @property
    def duration(self) -> float:
        """Total duration in beats."""
        if not self.chords:
            return 0
        return max(c.start_time + c.duration for c in self.chords)
    
    def to_symbols(self) -> List[str]:
        """Get list of chord symbols."""
        return [c.symbol for c in self.chords]


# Scale degree to chord quality mappings
MAJOR_SCALE_CHORDS = {
    "I": ("", "major"),
    "ii": ("", "minor"),
    "iii": ("", "minor"),
    "IV": ("", "major"),
    "V": ("", "major"),
    "vi": ("", "minor"),
    "vii": ("", "dim"),
    # Seventh chords
    "Imaj7": ("maj7", "maj7"),
    "ii7": ("7", "min7"),
    "iii7": ("7", "min7"),
    "IVmaj7": ("maj7", "maj7"),
    "V7": ("7", "7"),
    "vi7": ("7", "min7"),
    "vii7": ("7", "dim7"),
}

MINOR_SCALE_CHORDS = {
    "i": ("", "minor"),
    "ii": ("", "dim"),
    "III": ("", "major"),
    "iv": ("", "minor"),
    "v": ("", "minor"),
    "V": ("", "major"),  # Harmonic minor
    "VI": ("", "major"),
    "VII": ("", "major"),
    "vii": ("", "dim"),
}

# Scale degree to semitone offset
SCALE_DEGREES = {
    "I": 0, "i": 0,
    "II": 2, "ii": 2,
    "III": 4, "iii": 4,  # In major
    "IV": 5, "iv": 5,
    "V": 7, "v": 7,
    "VI": 9, "vi": 9,
    "VII": 11, "vii": 11,
}

# Adjust for minor scale
MINOR_SCALE_DEGREES = {
    "i": 0,
    "ii": 2,
    "III": 3,  # In minor
    "iv": 5,
    "v": 7,
    "V": 7,
    "VI": 8,
    "VII": 10,
    "vii": 11,
}


def generate_chords(
    melody_notes: List,
    key: str = "C",
    scale: str = "major",
    style: str = DEFAULT_STYLE,
    measures: int = None,
    beats_per_measure: int = 4,
    chords_per_measure: int = 1,
) -> ChordProgression:
    """
    Generate a chord progression that fits a melody.
    
    Args:
        melody_notes: List of Note objects
        key: Key root (e.g., "C", "G")
        scale: Scale type ("major" or "minor")
        style: Musical style ("pop", "jazz", "lofi", etc.)
        measures: Number of measures (auto-calculated if None)
        beats_per_measure: Beats per measure
        chords_per_measure: Number of chord changes per measure
    
    Returns:
        ChordProgression object
    """
    # Calculate number of measures from melody
    if measures is None:
        if melody_notes:
            total_beats = max(n.start_time + n.duration for n in melody_notes)
            measures = int(np.ceil(total_beats / beats_per_measure))
            measures = max(4, measures)  # Minimum 4 measures
        else:
            measures = 8
    
    # Get style-appropriate chord templates
    templates = CHORD_TEMPLATES.get(style, CHORD_TEMPLATES["pop"])
    
    # Analyze melody to find good chord choices
    melody_analysis = _analyze_melody_for_chords(
        melody_notes, key, scale, beats_per_measure
    )
    
    # Generate chord progression
    chords = _generate_progression(
        key=key,
        scale=scale,
        measures=measures,
        beats_per_measure=beats_per_measure,
        chords_per_measure=chords_per_measure,
        templates=templates,
        melody_analysis=melody_analysis,
    )
    
    return ChordProgression(chords=chords, key=key, scale=scale)


def _analyze_melody_for_chords(
    notes: List,
    key: str,
    scale: str,
    beats_per_measure: int,
) -> Dict:
    """
    Analyze melody to suggest chord choices.
    
    Returns dict mapping measure numbers to suggested scale degrees.
    """
    analysis = {}
    
    if not notes:
        return analysis
    
    # Get key root pitch class
    root_pc = NOTE_NAMES.index(key.upper()) if key.upper() in NOTE_NAMES else 0
    
    # Group notes by measure
    for note in notes:
        measure = int(note.start_time // beats_per_measure)
        beat_in_measure = note.start_time % beats_per_measure
        
        if measure not in analysis:
            analysis[measure] = {
                'notes': [],
                'strong_beat_notes': [],
                'pitch_classes': [],
            }
        
        # Get pitch class relative to key
        relative_pc = (note.pitch - root_pc) % 12
        
        analysis[measure]['notes'].append(note)
        analysis[measure]['pitch_classes'].append(relative_pc)
        
        # Notes on strong beats (1 and 3) are more important
        if beat_in_measure < 0.5 or (2 <= beat_in_measure < 2.5):
            analysis[measure]['strong_beat_notes'].append(relative_pc)
    
    # Suggest chords based on strong beat notes
    for measure, data in analysis.items():
        strong_notes = data['strong_beat_notes']
        if strong_notes:
            # Find chord that contains these notes
            data['suggested_degree'] = _suggest_chord_degree(strong_notes, scale)
        else:
            data['suggested_degree'] = None
    
    return analysis


def _suggest_chord_degree(pitch_classes: List[int], scale: str) -> str:
    """
    Suggest a scale degree based on melody pitch classes.
    """
    # Map pitch classes to scale degrees
    pc_to_degree = {
        0: "I" if scale == "major" else "i",
        2: "ii",
        4: "iii" if scale == "major" else "III",
        5: "IV" if scale == "major" else "iv",
        7: "V",
        9: "vi" if scale == "major" else "VI",
        11: "vii" if scale == "major" else "VII",
    }
    
    # Find most common pitch class
    if not pitch_classes:
        return "I" if scale == "major" else "i"
    
    from collections import Counter
    counts = Counter(pitch_classes)
    most_common_pc = counts.most_common(1)[0][0]
    
    # Return corresponding degree, or I/i as default
    return pc_to_degree.get(most_common_pc, "I" if scale == "major" else "i")


def _generate_progression(
    key: str,
    scale: str,
    measures: int,
    beats_per_measure: int,
    chords_per_measure: int,
    templates: List[List[str]],
    melody_analysis: Dict,
) -> List[Chord]:
    """
    Generate the actual chord progression.
    """
    chords = []
    
    # Choose a template
    template = random.choice(templates)
    template_len = len(template)
    
    # Chord duration
    chord_duration = beats_per_measure / chords_per_measure
    
    # Generate chords for each position
    chord_idx = 0
    for measure in range(measures):
        for chord_in_measure in range(chords_per_measure):
            # Get degree from template (cycling)
            degree = template[chord_idx % template_len]
            
            # Check if melody analysis suggests a different chord
            if measure in melody_analysis and melody_analysis[measure].get('suggested_degree'):
                # 50% chance to use melody-suggested chord
                if random.random() < 0.5:
                    degree = melody_analysis[measure]['suggested_degree']
            
            # Convert degree to actual chord
            chord = _degree_to_chord(key, scale, degree)
            
            # Set timing
            start_time = measure * beats_per_measure + chord_in_measure * chord_duration
            chord.start_time = start_time
            chord.duration = chord_duration
            
            chords.append(chord)
            chord_idx += 1
    
    return chords


def _degree_to_chord(key: str, scale: str, degree: str) -> Chord:
    """
    Convert a scale degree to a Chord object.
    
    Args:
        key: Key root (e.g., "C")
        scale: "major" or "minor"
        degree: Scale degree (e.g., "I", "ii", "V7")
    
    Returns:
        Chord object
    """
    # Get root pitch class
    key_pc = NOTE_NAMES.index(key.upper()) if key.upper() in NOTE_NAMES else 0
    
    # Parse degree (handle 7th chords)
    base_degree = degree.rstrip('7').rstrip('maj')
    has_7th = '7' in degree
    
    # Get semitone offset
    if scale == "minor" and base_degree in MINOR_SCALE_DEGREES:
        semitones = MINOR_SCALE_DEGREES[base_degree]
    else:
        semitones = SCALE_DEGREES.get(base_degree, 0)
    
    # Calculate chord root
    chord_root_pc = (key_pc + semitones) % 12
    chord_root = NOTE_NAMES[chord_root_pc]
    
    # Determine quality
    if scale == "major":
        chord_lookup = MAJOR_SCALE_CHORDS
    else:
        chord_lookup = MINOR_SCALE_CHORDS
    
    # Get quality from lookup, default to major
    if degree in chord_lookup:
        suffix, quality = chord_lookup[degree]
    elif base_degree in chord_lookup:
        suffix, quality = chord_lookup[base_degree]
        if has_7th:
            quality = quality.replace('major', 'maj7').replace('minor', 'min7')
            suffix = '7' if 'min' not in quality and 'maj' not in quality else suffix
    else:
        suffix = ""
        quality = "major"
    
    # Build chord symbol
    symbol = chord_root + suffix
    
    return Chord(
        root=chord_root,
        quality=quality,
        symbol=symbol,
        start_time=0,
        duration=4,
    )


def harmonize_melody(
    melody_notes: List,
    key: str = "C",
    scale: str = "major",
) -> List[Chord]:
    """
    Generate a chord for each melody note (simple harmonization).
    
    Args:
        melody_notes: List of Note objects
        key: Key root
        scale: Scale type
    
    Returns:
        List of Chord objects
    """
    chords = []
    key_pc = NOTE_NAMES.index(key.upper()) if key.upper() in NOTE_NAMES else 0
    
    for note in melody_notes:
        # Get melody note's scale degree
        relative_pc = (note.pitch - key_pc) % 12
        
        # Choose chord containing this note
        degree = _suggest_chord_degree([relative_pc], scale)
        
        chord = _degree_to_chord(key, scale, degree)
        chord.start_time = note.start_time
        chord.duration = note.duration
        
        chords.append(chord)
    
    return chords

