"""
Harmony generation module for HumToHarmony.

Generates chord progressions and bass lines to accompany melodies.
"""

from .chord_generator import (
    generate_chords,
    ChordProgression,
    Chord,
)
from .bass_generator import (
    generate_bass_line,
)
from .voicing import (
    voice_chord,
    create_chord_voicing,
)
from .styles import (
    get_style_settings,
    STYLE_SETTINGS,
)

__all__ = [
    "generate_chords",
    "ChordProgression", 
    "Chord",
    "generate_bass_line",
    "voice_chord",
    "create_chord_voicing",
    "get_style_settings",
    "STYLE_SETTINGS",
]

