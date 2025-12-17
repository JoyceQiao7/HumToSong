"""
Utility functions for HumToHarmony.
"""

from .audio_io import (
    load_audio,
    save_audio,
    normalize_audio,
    trim_silence,
    get_duration,
    resample,
)
from .midi_utils import (
    note_to_midi,
    midi_to_note,
    midi_to_freq,
    freq_to_midi,
    NOTE_NAMES,
)
from .visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_piano_roll,
)

__all__ = [
    # Audio I/O
    "load_audio",
    "save_audio", 
    "normalize_audio",
    "trim_silence",
    "get_duration",
    "resample",
    # MIDI
    "note_to_midi",
    "midi_to_note",
    "midi_to_freq",
    "freq_to_midi",
    "NOTE_NAMES",
    # Visualization
    "plot_waveform",
    "plot_spectrogram",
    "plot_piano_roll",
]

