"""
Timbre descriptor vocabulary for HumToHarmony.

Maps descriptive words to synthesizer parameters.
"""

from typing import Dict, Any, List

# Comprehensive mapping of descriptive terms to synth parameters
TIMBRE_DESCRIPTORS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # TONE COLOR / BRIGHTNESS
    # ==========================================================================
    "bright": {
        "filter_cutoff": 0.85,
        "filter_resonance": 0.3,
        "high_boost": 3,
    },
    "dark": {
        "filter_cutoff": 0.3,
        "filter_resonance": 0.2,
        "low_boost": 3,
    },
    "warm": {
        "filter_cutoff": 0.5,
        "waveform": "sawtooth",
        "harmonics": "even",
        "low_boost": 2,
    },
    "cold": {
        "filter_cutoff": 0.7,
        "waveform": "square",
        "filter_resonance": 0.4,
    },
    "harsh": {
        "filter_cutoff": 0.9,
        "filter_resonance": 0.6,
        "distortion": 0.3,
    },
    "mellow": {
        "filter_cutoff": 0.4,
        "filter_resonance": 0.1,
        "waveform": "sine",
    },
    "crisp": {
        "filter_cutoff": 0.8,
        "attack": 0.001,
        "filter_envelope": 0.5,
    },
    "dull": {
        "filter_cutoff": 0.25,
        "filter_resonance": 0.0,
    },
    
    # ==========================================================================
    # CHARACTER / TEXTURE
    # ==========================================================================
    "analog": {
        "detune": 5,
        "noise_amount": 0.02,
        "filter_resonance": 0.3,
        "drift": True,
    },
    "digital": {
        "detune": 0,
        "noise_amount": 0.0,
        "waveform": "sawtooth",
    },
    "vintage": {
        "filter_cutoff": 0.5,
        "saturation": 0.2,
        "low_boost": 2,
        "high_cut": True,
    },
    "modern": {
        "filter_cutoff": 0.7,
        "unison_voices": 2,
        "stereo_width": 0.8,
    },
    "retro": {
        "filter_cutoff": 0.45,
        "waveform": "square",
        "filter_resonance": 0.35,
    },
    "futuristic": {
        "unison_voices": 4,
        "unison_spread": 25,
        "filter_resonance": 0.5,
    },
    "organic": {
        "noise_amount": 0.03,
        "filter_cutoff": 0.5,
        "attack": 0.05,
    },
    "synthetic": {
        "waveform": "sawtooth",
        "unison_voices": 3,
        "filter_resonance": 0.4,
    },
    "clean": {
        "distortion": 0.0,
        "noise_amount": 0.0,
        "filter_resonance": 0.1,
    },
    "dirty": {
        "distortion": 0.3,
        "noise_amount": 0.05,
        "saturation": 0.3,
    },
    "gritty": {
        "distortion": 0.4,
        "noise_amount": 0.03,
        "filter_cutoff": 0.6,
    },
    "smooth": {
        "attack": 0.05,
        "filter_resonance": 0.1,
        "filter_cutoff": 0.5,
    },
    "rough": {
        "noise_amount": 0.04,
        "distortion": 0.2,
    },
    
    # ==========================================================================
    # DYNAMICS / ENVELOPE
    # ==========================================================================
    "punchy": {
        "attack": 0.001,
        "decay": 0.15,
        "sustain": 0.4,
        "filter_envelope": 0.5,
    },
    "soft": {
        "attack": 0.1,
        "velocity_sensitivity": 0.3,
        "filter_cutoff": 0.4,
    },
    "aggressive": {
        "attack": 0.001,
        "distortion": 0.3,
        "filter_resonance": 0.5,
    },
    "gentle": {
        "attack": 0.15,
        "filter_cutoff": 0.4,
        "release": 0.5,
    },
    "snappy": {
        "attack": 0.001,
        "decay": 0.1,
        "filter_envelope": 0.6,
    },
    "sustained": {
        "sustain": 0.9,
        "release": 0.5,
    },
    "short": {
        "decay": 0.1,
        "sustain": 0.0,
        "release": 0.1,
    },
    "long": {
        "sustain": 0.8,
        "release": 1.0,
    },
    "plucky": {
        "attack": 0.001,
        "decay": 0.3,
        "sustain": 0.0,
        "filter_envelope": 0.6,
    },
    
    # ==========================================================================
    # SPACE / EFFECTS
    # ==========================================================================
    "wet": {
        "reverb": 0.6,
        "delay": 0.3,
    },
    "dry": {
        "reverb": 0.0,
        "delay": 0.0,
    },
    "spacious": {
        "reverb": 0.5,
        "stereo_width": 1.0,
    },
    "intimate": {
        "reverb": 0.15,
        "proximity": True,
    },
    "dreamy": {
        "reverb": 0.7,
        "delay": 0.4,
        "filter_cutoff": 0.5,
        "attack": 0.2,
    },
    "ethereal": {
        "reverb": 0.8,
        "shimmer": True,
        "attack": 0.3,
        "filter_cutoff": 0.6,
    },
    "airy": {
        "reverb": 0.5,
        "high_boost": 2,
        "filter_cutoff": 0.7,
    },
    "thick": {
        "unison_voices": 4,
        "unison_spread": 20,
        "low_boost": 3,
    },
    "thin": {
        "unison_voices": 1,
        "filter_cutoff": 0.7,
        "low_cut": True,
    },
    "wide": {
        "stereo_width": 1.0,
        "unison_voices": 3,
        "unison_spread": 30,
    },
    "narrow": {
        "stereo_width": 0.2,
        "unison_voices": 1,
    },
    
    # ==========================================================================
    # INSTRUMENT TYPES (base presets)
    # ==========================================================================
    "piano": {
        "preset": "piano",
        "waveform": "triangle",
        "attack": 0.01,
        "decay": 0.5,
        "sustain": 0.3,
    },
    "synth": {
        "preset": "synth",
        "waveform": "sawtooth",
        "unison_voices": 2,
    },
    "organ": {
        "preset": "organ",
        "waveform": "sine",
        "attack": 0.001,
        "sustain": 1.0,
    },
    "strings": {
        "preset": "strings",
        "waveform": "sawtooth",
        "attack": 0.2,
        "unison_voices": 4,
    },
    "pad": {
        "preset": "pad",
        "attack": 0.5,
        "sustain": 0.8,
        "release": 1.0,
    },
    "lead": {
        "preset": "lead",
        "waveform": "sawtooth",
        "unison_voices": 2,
        "mono": True,
    },
    "bass": {
        "preset": "bass",
        "filter_cutoff": 0.35,
        "octave_shift": -1,
    },
    "bells": {
        "preset": "bells",
        "waveform": "sine",
        "fm_ratio": 1.4,
        "attack": 0.001,
        "decay": 1.0,
    },
    "keys": {
        "preset": "keys",
        "waveform": "triangle",
        "attack": 0.01,
    },
    "rhodes": {
        "preset": "rhodes",
        "waveform": "sine",
        "fm_ratio": 1.0,
        "tremolo": True,
    },
    "electric": {
        "preset": "electric",
        "distortion": 0.1,
        "filter_resonance": 0.3,
    },
    "acoustic": {
        "preset": "acoustic",
        "noise_amount": 0.02,
        "filter_cutoff": 0.6,
    },
    
    # ==========================================================================
    # REGISTER / RANGE
    # ==========================================================================
    "deep": {
        "octave_shift": -1,
        "low_boost": 4,
    },
    "high": {
        "octave_shift": 1,
        "high_boost": 2,
    },
    "sub": {
        "octave_shift": -2,
        "filter_cutoff": 0.2,
    },
    
    # ==========================================================================
    # WAVEFORM HINTS
    # ==========================================================================
    "buzzy": {
        "waveform": "sawtooth",
        "filter_resonance": 0.4,
    },
    "hollow": {
        "waveform": "square",
        "filter_type": "bandpass",
    },
    "pure": {
        "waveform": "sine",
        "filter_cutoff": 0.9,
    },
    "rich": {
        "waveform": "sawtooth",
        "unison_voices": 3,
        "filter_cutoff": 0.7,
    },
}


def get_descriptor_mapping(descriptor: str) -> Dict[str, Any]:
    """
    Get parameter mapping for a descriptor.
    
    Args:
        descriptor: Descriptive word (e.g., "warm", "bright")
    
    Returns:
        Dictionary of synth parameters
    """
    descriptor_lower = descriptor.lower().strip()
    return TIMBRE_DESCRIPTORS.get(descriptor_lower, {})


def get_all_descriptors() -> List[str]:
    """Get list of all known descriptors."""
    return list(TIMBRE_DESCRIPTORS.keys())


def get_descriptors_by_category() -> Dict[str, List[str]]:
    """Get descriptors organized by category."""
    categories = {
        "Tone Color": ["bright", "dark", "warm", "cold", "harsh", "mellow", "crisp", "dull"],
        "Character": ["analog", "digital", "vintage", "modern", "clean", "dirty", "gritty", "smooth"],
        "Dynamics": ["punchy", "soft", "aggressive", "gentle", "snappy", "plucky", "sustained"],
        "Space": ["wet", "dry", "spacious", "dreamy", "ethereal", "airy", "thick", "thin", "wide"],
        "Instruments": ["piano", "synth", "organ", "strings", "pad", "lead", "bass", "bells", "rhodes"],
        "Register": ["deep", "high", "sub"],
    }
    return categories

