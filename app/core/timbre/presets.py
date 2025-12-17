"""
Sound presets for HumToHarmony.

Provides quick preset sounds for common instrument types.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.synth.engine import SynthParams


@dataclass
class Preset:
    """A sound preset."""
    name: str
    category: str
    description: str
    synth_params: SynthParams
    effects: Dict[str, float]
    tags: List[str]


# Comprehensive preset library
PRESETS: Dict[str, Preset] = {
    # ==========================================================================
    # PIANO / KEYS
    # ==========================================================================
    "grand_piano": Preset(
        name="Grand Piano",
        category="keys",
        description="Classic acoustic grand piano sound",
        synth_params=SynthParams(
            waveform="triangle",
            attack=0.01,
            decay=0.8,
            sustain=0.3,
            release=0.4,
            filter_cutoff=0.65,
            filter_envelope=0.3,
        ),
        effects={"reverb": 0.3, "delay": 0.0, "chorus": 0.0},
        tags=["piano", "acoustic", "classic", "warm"],
    ),
    
    "bright_piano": Preset(
        name="Bright Piano",
        category="keys",
        description="Bright, modern piano sound",
        synth_params=SynthParams(
            waveform="triangle",
            attack=0.005,
            decay=0.6,
            sustain=0.25,
            release=0.3,
            filter_cutoff=0.8,
            filter_envelope=0.4,
        ),
        effects={"reverb": 0.25, "delay": 0.0, "chorus": 0.0},
        tags=["piano", "bright", "modern"],
    ),
    
    "rhodes": Preset(
        name="Rhodes EP",
        category="keys",
        description="Warm Fender Rhodes electric piano",
        synth_params=SynthParams(
            waveform="sine",
            osc2_enabled=True,
            osc2_waveform="sine",
            osc2_mix=0.3,
            attack=0.01,
            decay=0.6,
            sustain=0.4,
            release=0.5,
            filter_cutoff=0.5,
            filter_resonance=0.2,
        ),
        effects={"reverb": 0.25, "delay": 0.1, "chorus": 0.3},
        tags=["rhodes", "electric", "warm", "vintage", "keys"],
    ),
    
    "wurlitzer": Preset(
        name="Wurlitzer",
        category="keys",
        description="Classic Wurlitzer electric piano",
        synth_params=SynthParams(
            waveform="triangle",
            attack=0.01,
            decay=0.4,
            sustain=0.3,
            release=0.3,
            filter_cutoff=0.55,
            distortion=0.1,
        ),
        effects={"reverb": 0.2, "delay": 0.0, "chorus": 0.2},
        tags=["wurlitzer", "electric", "vintage", "keys"],
    ),
    
    # ==========================================================================
    # SYNTH LEADS
    # ==========================================================================
    "classic_lead": Preset(
        name="Classic Synth Lead",
        category="lead",
        description="Classic sawtooth synth lead",
        synth_params=SynthParams(
            waveform="sawtooth",
            unison_voices=2,
            unison_spread=12,
            attack=0.01,
            decay=0.2,
            sustain=0.7,
            release=0.3,
            filter_cutoff=0.7,
            filter_resonance=0.3,
        ),
        effects={"reverb": 0.2, "delay": 0.15, "chorus": 0.0},
        tags=["synth", "lead", "classic", "bright"],
    ),
    
    "soft_lead": Preset(
        name="Soft Lead",
        category="lead",
        description="Gentle, soft synth lead",
        synth_params=SynthParams(
            waveform="triangle",
            attack=0.05,
            decay=0.3,
            sustain=0.6,
            release=0.4,
            filter_cutoff=0.5,
        ),
        effects={"reverb": 0.3, "delay": 0.1, "chorus": 0.2},
        tags=["synth", "lead", "soft", "gentle"],
    ),
    
    "supersaw_lead": Preset(
        name="Supersaw Lead",
        category="lead",
        description="Thick, modern supersaw lead",
        synth_params=SynthParams(
            waveform="sawtooth",
            unison_voices=5,
            unison_spread=25,
            attack=0.01,
            decay=0.2,
            sustain=0.8,
            release=0.3,
            filter_cutoff=0.75,
            filter_resonance=0.25,
        ),
        effects={"reverb": 0.25, "delay": 0.0, "chorus": 0.0},
        tags=["synth", "lead", "supersaw", "thick", "modern"],
    ),
    
    # ==========================================================================
    # PADS
    # ==========================================================================
    "warm_pad": Preset(
        name="Warm Pad",
        category="pad",
        description="Warm, enveloping pad sound",
        synth_params=SynthParams(
            waveform="sawtooth",
            unison_voices=4,
            unison_spread=15,
            attack=0.5,
            decay=0.3,
            sustain=0.8,
            release=1.0,
            filter_cutoff=0.45,
            filter_resonance=0.15,
        ),
        effects={"reverb": 0.5, "delay": 0.2, "chorus": 0.3},
        tags=["pad", "warm", "soft", "ambient"],
    ),
    
    "ethereal_pad": Preset(
        name="Ethereal Pad",
        category="pad",
        description="Dreamy, ethereal pad",
        synth_params=SynthParams(
            waveform="sawtooth",
            unison_voices=4,
            unison_spread=20,
            attack=0.8,
            decay=0.5,
            sustain=0.7,
            release=1.5,
            filter_cutoff=0.5,
        ),
        effects={"reverb": 0.7, "delay": 0.35, "chorus": 0.2},
        tags=["pad", "ethereal", "dreamy", "spacious"],
    ),
    
    "dark_pad": Preset(
        name="Dark Pad",
        category="pad",
        description="Dark, moody pad",
        synth_params=SynthParams(
            waveform="sawtooth",
            unison_voices=3,
            unison_spread=10,
            attack=0.6,
            decay=0.4,
            sustain=0.7,
            release=1.0,
            filter_cutoff=0.3,
            filter_resonance=0.2,
        ),
        effects={"reverb": 0.4, "delay": 0.15, "chorus": 0.1},
        tags=["pad", "dark", "moody", "atmospheric"],
    ),
    
    # ==========================================================================
    # BASS
    # ==========================================================================
    "sub_bass": Preset(
        name="Sub Bass",
        category="bass",
        description="Deep sub bass",
        synth_params=SynthParams(
            waveform="sine",
            attack=0.01,
            decay=0.1,
            sustain=0.9,
            release=0.2,
            filter_cutoff=0.25,
        ),
        effects={"reverb": 0.0, "delay": 0.0, "chorus": 0.0},
        tags=["bass", "sub", "deep", "clean"],
    ),
    
    "synth_bass": Preset(
        name="Synth Bass",
        category="bass",
        description="Classic synth bass",
        synth_params=SynthParams(
            waveform="sawtooth",
            attack=0.01,
            decay=0.15,
            sustain=0.7,
            release=0.2,
            filter_cutoff=0.4,
            filter_resonance=0.35,
            filter_envelope=0.4,
        ),
        effects={"reverb": 0.0, "delay": 0.0, "chorus": 0.0},
        tags=["bass", "synth", "punchy"],
    ),
    
    "wobble_bass": Preset(
        name="Wobble Bass",
        category="bass",
        description="Dubstep-style wobble bass",
        synth_params=SynthParams(
            waveform="sawtooth",
            osc2_enabled=True,
            osc2_waveform="square",
            osc2_mix=0.4,
            attack=0.01,
            decay=0.1,
            sustain=0.8,
            release=0.1,
            filter_cutoff=0.5,
            filter_resonance=0.5,
            filter_envelope=0.6,
        ),
        effects={"reverb": 0.1, "delay": 0.0, "chorus": 0.0},
        tags=["bass", "wobble", "dubstep", "electronic"],
    ),
    
    "acoustic_bass": Preset(
        name="Acoustic Bass",
        category="bass",
        description="Upright/acoustic bass sound",
        synth_params=SynthParams(
            waveform="triangle",
            noise_amount=0.02,
            attack=0.02,
            decay=0.3,
            sustain=0.5,
            release=0.3,
            filter_cutoff=0.45,
        ),
        effects={"reverb": 0.2, "delay": 0.0, "chorus": 0.0},
        tags=["bass", "acoustic", "upright", "warm"],
    ),
    
    # ==========================================================================
    # STRINGS
    # ==========================================================================
    "string_ensemble": Preset(
        name="String Ensemble",
        category="strings",
        description="Lush orchestral strings",
        synth_params=SynthParams(
            waveform="sawtooth",
            unison_voices=4,
            unison_spread=12,
            attack=0.25,
            decay=0.2,
            sustain=0.75,
            release=0.5,
            filter_cutoff=0.55,
        ),
        effects={"reverb": 0.45, "delay": 0.0, "chorus": 0.2},
        tags=["strings", "orchestral", "lush", "warm"],
    ),
    
    "solo_violin": Preset(
        name="Solo Violin",
        category="strings",
        description="Expressive solo violin",
        synth_params=SynthParams(
            waveform="sawtooth",
            attack=0.15,
            decay=0.2,
            sustain=0.8,
            release=0.3,
            filter_cutoff=0.6,
            filter_resonance=0.2,
        ),
        effects={"reverb": 0.35, "delay": 0.0, "chorus": 0.15},
        tags=["strings", "violin", "solo", "expressive"],
    ),
    
    # ==========================================================================
    # PLUCKS
    # ==========================================================================
    "pluck": Preset(
        name="Synth Pluck",
        category="pluck",
        description="Short, punchy pluck sound",
        synth_params=SynthParams(
            waveform="sawtooth",
            attack=0.001,
            decay=0.25,
            sustain=0.0,
            release=0.15,
            filter_cutoff=0.75,
            filter_envelope=0.5,
            filter_decay=0.2,
            filter_sustain=0.2,
        ),
        effects={"reverb": 0.25, "delay": 0.2, "chorus": 0.0},
        tags=["pluck", "short", "punchy", "synth"],
    ),
    
    "guitar_pluck": Preset(
        name="Acoustic Guitar",
        category="pluck",
        description="Acoustic guitar-like pluck",
        synth_params=SynthParams(
            waveform="triangle",
            noise_amount=0.03,
            attack=0.001,
            decay=0.4,
            sustain=0.2,
            release=0.3,
            filter_cutoff=0.6,
            filter_envelope=0.4,
        ),
        effects={"reverb": 0.2, "delay": 0.0, "chorus": 0.1},
        tags=["guitar", "acoustic", "pluck"],
    ),
}


def get_preset(name: str) -> Preset:
    """
    Get a preset by name.
    
    Args:
        name: Preset name (case-insensitive)
    
    Returns:
        Preset object or default if not found
    """
    name_lower = name.lower().replace(" ", "_")
    
    if name_lower in PRESETS:
        return PRESETS[name_lower]
    
    # Try to find partial match
    for key, preset in PRESETS.items():
        if name_lower in key or name_lower in preset.name.lower():
            return preset
    
    # Default to classic lead
    return PRESETS.get("classic_lead", list(PRESETS.values())[0])


def get_preset_names() -> List[str]:
    """Get all preset names."""
    return [p.name for p in PRESETS.values()]


def get_presets_by_category() -> Dict[str, List[str]]:
    """Get preset names organized by category."""
    by_category = {}
    
    for preset in PRESETS.values():
        if preset.category not in by_category:
            by_category[preset.category] = []
        by_category[preset.category].append(preset.name)
    
    return by_category


def search_presets(query: str) -> List[Preset]:
    """
    Search presets by name, description, or tags.
    
    Args:
        query: Search query
    
    Returns:
        List of matching presets
    """
    query_lower = query.lower()
    matches = []
    
    for preset in PRESETS.values():
        # Check name
        if query_lower in preset.name.lower():
            matches.append(preset)
            continue
        
        # Check description
        if query_lower in preset.description.lower():
            matches.append(preset)
            continue
        
        # Check tags
        if any(query_lower in tag for tag in preset.tags):
            matches.append(preset)
    
    return matches

