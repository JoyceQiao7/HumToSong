"""
Sound synthesis module for HumToHarmony.

Provides a flexible synthesizer engine with:
- Multiple oscillator types
- Filters and envelopes
- Effects processing
"""

from .oscillators import (
    sine_wave,
    triangle_wave,
    sawtooth_wave,
    square_wave,
    noise,
    Oscillator,
)
from .envelopes import (
    ADSREnvelope,
    apply_envelope,
)
from .filters import (
    lowpass_filter,
    highpass_filter,
    bandpass_filter,
)
from .engine import (
    Synthesizer,
    SynthParams,
    render_notes,
)

__all__ = [
    # Oscillators
    "sine_wave",
    "triangle_wave", 
    "sawtooth_wave",
    "square_wave",
    "noise",
    "Oscillator",
    # Envelopes
    "ADSREnvelope",
    "apply_envelope",
    # Filters
    "lowpass_filter",
    "highpass_filter",
    "bandpass_filter",
    # Engine
    "Synthesizer",
    "SynthParams",
    "render_notes",
]

