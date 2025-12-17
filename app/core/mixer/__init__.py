"""
Audio mixing and effects module for HumToHarmony.

Provides:
- Audio effects (reverb, delay, chorus)
- Track mixing
- Master processing
- Export functionality
"""

from .effects import (
    apply_reverb,
    apply_delay,
    apply_chorus,
    apply_compression,
    apply_limiter,
    EffectsChain,
)
from .tracks import (
    Track,
    mix_tracks,
)
from .export import (
    export_wav,
    export_mp3,
    export_midi,
)

__all__ = [
    # Effects
    "apply_reverb",
    "apply_delay",
    "apply_chorus",
    "apply_compression",
    "apply_limiter",
    "EffectsChain",
    # Tracks
    "Track",
    "mix_tracks",
    # Export
    "export_wav",
    "export_mp3",
    "export_midi",
]

