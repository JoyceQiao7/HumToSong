"""
Natural language timbre system for HumToHarmony.

Converts natural language descriptions into synthesizer parameters.
"""

from .descriptors import (
    TIMBRE_DESCRIPTORS,
    get_descriptor_mapping,
)
from .parser import (
    parse_timbre_description,
    extract_descriptors,
)
from .mapper import (
    description_to_params,
    TimbreMapper,
)
from .presets import (
    get_preset,
    get_preset_names,
    PRESETS,
)

__all__ = [
    "TIMBRE_DESCRIPTORS",
    "get_descriptor_mapping",
    "parse_timbre_description",
    "extract_descriptors",
    "description_to_params",
    "TimbreMapper",
    "get_preset",
    "get_preset_names",
    "PRESETS",
]

