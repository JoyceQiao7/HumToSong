"""
Pitch detection and note extraction module.

Uses CREPE for accurate monophonic pitch detection,
then converts pitch contours to discrete notes.
"""

from .detector import (
    detect_pitch,
    extract_notes,
    PitchResult,
)
from .quantizer import (
    quantize_notes,
    detect_tempo,
)
from .key_detector import (
    detect_key,
    KeyResult,
)

__all__ = [
    "detect_pitch",
    "extract_notes",
    "PitchResult",
    "quantize_notes",
    "detect_tempo",
    "detect_key",
    "KeyResult",
]

