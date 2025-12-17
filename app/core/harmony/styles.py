"""
Musical style configurations for HumToHarmony.

Defines parameters for different musical styles affecting:
- Chord progressions
- Bass patterns
- Voicings
- Rhythm
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StyleSettings:
    """Configuration for a musical style."""
    name: str
    
    # Chord settings
    chord_complexity: str = "simple"  # simple, moderate, complex
    prefer_7ths: bool = False
    common_progressions: List[List[str]] = field(default_factory=list)
    
    # Bass settings
    bass_pattern: str = "quarter"  # quarter, eighth, walking, sparse
    bass_octave: int = 2
    
    # Harmony settings
    voicing_type: str = "close"  # close, open, spread, drop2
    harmony_rhythm: str = "whole"  # whole, half, quarter
    
    # Rhythm settings
    default_tempo: float = 120.0
    swing: float = 0.0  # 0 = straight, 0.5 = heavy swing
    
    # Timbre suggestions
    melody_timbre: str = "piano"
    harmony_timbre: str = "piano"
    bass_timbre: str = "bass"
    
    # Effects
    reverb_amount: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'chord_complexity': self.chord_complexity,
            'prefer_7ths': self.prefer_7ths,
            'bass_pattern': self.bass_pattern,
            'voicing_type': self.voicing_type,
            'default_tempo': self.default_tempo,
            'swing': self.swing,
        }


# Predefined style configurations
STYLE_SETTINGS = {
    "pop": StyleSettings(
        name="Pop",
        chord_complexity="simple",
        prefer_7ths=False,
        common_progressions=[
            ["I", "V", "vi", "IV"],
            ["I", "IV", "V", "I"],
            ["vi", "IV", "I", "V"],
            ["I", "vi", "IV", "V"],
        ],
        bass_pattern="quarter",
        bass_octave=2,
        voicing_type="close",
        harmony_rhythm="whole",
        default_tempo=120.0,
        swing=0.0,
        melody_timbre="bright synth lead",
        harmony_timbre="warm piano",
        bass_timbre="deep sub bass",
        reverb_amount=0.25,
    ),
    
    "jazz": StyleSettings(
        name="Jazz",
        chord_complexity="complex",
        prefer_7ths=True,
        common_progressions=[
            ["ii7", "V7", "Imaj7", "Imaj7"],
            ["Imaj7", "vi7", "ii7", "V7"],
            ["iii7", "vi7", "ii7", "V7"],
            ["Imaj7", "IVmaj7", "iii7", "vi7"],
        ],
        bass_pattern="walking",
        bass_octave=2,
        voicing_type="drop2",
        harmony_rhythm="half",
        default_tempo=140.0,
        swing=0.3,
        melody_timbre="mellow saxophone",
        harmony_timbre="rhodes electric piano",
        bass_timbre="upright bass",
        reverb_amount=0.3,
    ),
    
    "lofi": StyleSettings(
        name="Lo-Fi",
        chord_complexity="moderate",
        prefer_7ths=True,
        common_progressions=[
            ["ii7", "V7", "Imaj7", "vi7"],
            ["Imaj7", "V7", "vi7", "iii7"],
            ["vi7", "ii7", "V7", "Imaj7"],
        ],
        bass_pattern="sparse",
        bass_octave=2,
        voicing_type="spread",
        harmony_rhythm="whole",
        default_tempo=85.0,
        swing=0.15,
        melody_timbre="warm vintage keys",
        harmony_timbre="dusty rhodes",
        bass_timbre="mellow sub bass",
        reverb_amount=0.4,
    ),
    
    "classical": StyleSettings(
        name="Classical",
        chord_complexity="moderate",
        prefer_7ths=False,
        common_progressions=[
            ["I", "IV", "V", "I"],
            ["I", "ii", "V", "I"],
            ["I", "vi", "IV", "V"],
            ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
        ],
        bass_pattern="quarter",
        bass_octave=3,
        voicing_type="close",
        harmony_rhythm="half",
        default_tempo=100.0,
        swing=0.0,
        melody_timbre="strings violin",
        harmony_timbre="strings ensemble",
        bass_timbre="cello bass",
        reverb_amount=0.5,
    ),
    
    "electronic": StyleSettings(
        name="Electronic",
        chord_complexity="simple",
        prefer_7ths=False,
        common_progressions=[
            ["i", "VI", "III", "VII"],
            ["i", "iv", "VI", "V"],
            ["vi", "IV", "I", "V"],
            ["i", "i", "VI", "VII"],
        ],
        bass_pattern="eighth",
        bass_octave=1,
        voicing_type="open",
        harmony_rhythm="whole",
        default_tempo=128.0,
        swing=0.0,
        melody_timbre="bright digital lead",
        harmony_timbre="wide supersaw pad",
        bass_timbre="aggressive sub bass",
        reverb_amount=0.35,
    ),
}


def get_style_settings(style_name: str) -> StyleSettings:
    """
    Get settings for a musical style.
    
    Args:
        style_name: Name of the style (pop, jazz, lofi, classical, electronic)
    
    Returns:
        StyleSettings object
    """
    style_lower = style_name.lower()
    
    if style_lower in STYLE_SETTINGS:
        return STYLE_SETTINGS[style_lower]
    
    # Default to pop
    return STYLE_SETTINGS["pop"]


def get_available_styles() -> List[str]:
    """Get list of available style names."""
    return list(STYLE_SETTINGS.keys())


def get_style_description(style_name: str) -> str:
    """Get a human-readable description of a style."""
    descriptions = {
        "pop": "Modern pop with simple chord progressions and driving rhythms",
        "jazz": "Jazz with extended harmonies, walking bass, and swing feel",
        "lofi": "Chill lo-fi hip hop with jazzy chords and relaxed groove",
        "classical": "Classical-inspired with traditional harmony and orchestral sounds",
        "electronic": "Electronic/EDM with powerful bass and atmospheric pads",
    }
    return descriptions.get(style_name.lower(), "Unknown style")

