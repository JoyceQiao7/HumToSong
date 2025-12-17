"""
Maps natural language descriptions to synthesizer parameters.
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.synth.engine import SynthParams
from .descriptors import get_descriptor_mapping
from .parser import parse_timbre_description


@dataclass
class TimbreMapping:
    """Result of timbre description mapping."""
    synth_params: SynthParams
    descriptors_used: List[str]
    effects_params: Dict[str, float]
    confidence: float


class TimbreMapper:
    """
    Maps natural language timbre descriptions to synth parameters.
    """
    
    def __init__(self):
        # Default base parameters
        self.base_params = {
            "waveform": "sawtooth",
            "detune": 0.0,
            "unison_voices": 1,
            "unison_spread": 10.0,
            "filter_cutoff": 0.6,
            "filter_resonance": 0.2,
            "filter_envelope": 0.2,
            "attack": 0.01,
            "decay": 0.2,
            "sustain": 0.7,
            "release": 0.3,
            "distortion": 0.0,
            "noise_amount": 0.0,
            "gain": 0.8,
        }
        
        # Effects defaults
        self.base_effects = {
            "reverb": 0.25,
            "delay": 0.0,
            "chorus": 0.0,
        }
    
    def map_description(self, description: str) -> TimbreMapping:
        """
        Map a natural language description to synth parameters.
        
        Args:
            description: User's timbre description
        
        Returns:
            TimbreMapping with synth params and effects
        """
        # Parse description
        parsed = parse_timbre_description(description)
        
        if not parsed:
            # Return defaults
            return TimbreMapping(
                synth_params=SynthParams(**self.base_params),
                descriptors_used=[],
                effects_params=self.base_effects.copy(),
                confidence=0.0,
            )
        
        # Start with base parameters
        params = self.base_params.copy()
        effects = self.base_effects.copy()
        descriptors_used = []
        
        # Apply each descriptor
        for descriptor, weight in parsed:
            mapping = get_descriptor_mapping(descriptor)
            
            if mapping:
                descriptors_used.append(descriptor)
                
                for key, value in mapping.items():
                    # Skip non-synth params
                    if key in ["preset", "shimmer", "drift", "tremolo", 
                               "proximity", "high_cut", "low_cut", "mono",
                               "fm_ratio", "velocity_sensitivity", "harmonics"]:
                        continue
                    
                    # Handle effects separately
                    if key in ["reverb", "delay", "chorus"]:
                        effects[key] = self._blend_value(
                            effects.get(key, 0),
                            value,
                            weight
                        )
                        continue
                    
                    # Handle EQ params
                    if key in ["low_boost", "high_boost", "low_cut", "high_cut"]:
                        # Store for effects processing
                        effects[key] = value
                        continue
                    
                    # Handle octave shift
                    if key == "octave_shift":
                        # Will be handled in note rendering
                        effects["octave_shift"] = value
                        continue
                    
                    # Handle stereo width
                    if key == "stereo_width":
                        effects["stereo_width"] = value
                        continue
                    
                    # Handle saturation
                    if key == "saturation":
                        params["distortion"] = self._blend_value(
                            params.get("distortion", 0),
                            value * 0.5,  # Saturation is gentler than distortion
                            weight
                        )
                        continue
                    
                    # Blend parameter value
                    if key in params:
                        current = params[key]
                        if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                            params[key] = self._blend_value(current, value, weight)
                        else:
                            # For non-numeric (like waveform), just replace
                            params[key] = value
        
        # Clamp values to valid ranges
        params = self._clamp_params(params)
        
        # Calculate confidence based on how many descriptors were recognized
        confidence = min(1.0, len(descriptors_used) / 3.0)
        
        return TimbreMapping(
            synth_params=SynthParams(**params),
            descriptors_used=descriptors_used,
            effects_params=effects,
            confidence=confidence,
        )
    
    def _blend_value(self, current: float, new: float, weight: float) -> float:
        """Blend two values based on weight."""
        # Weight affects how much the new value influences the result
        blend = 0.5 * weight
        return current * (1 - blend) + new * blend
    
    def _clamp_params(self, params: dict) -> dict:
        """Clamp parameters to valid ranges."""
        clamped = params.copy()
        
        # Filter cutoff: 0-1
        if "filter_cutoff" in clamped:
            clamped["filter_cutoff"] = max(0.05, min(0.99, clamped["filter_cutoff"]))
        
        # Filter resonance: 0-0.9
        if "filter_resonance" in clamped:
            clamped["filter_resonance"] = max(0.0, min(0.9, clamped["filter_resonance"]))
        
        # Envelope times: positive
        for env_param in ["attack", "decay", "release"]:
            if env_param in clamped:
                clamped[env_param] = max(0.001, min(5.0, clamped[env_param]))
        
        # Sustain: 0-1
        if "sustain" in clamped:
            clamped["sustain"] = max(0.0, min(1.0, clamped["sustain"]))
        
        # Distortion: 0-1
        if "distortion" in clamped:
            clamped["distortion"] = max(0.0, min(0.9, clamped["distortion"]))
        
        # Unison voices: 1-8
        if "unison_voices" in clamped:
            clamped["unison_voices"] = max(1, min(8, int(clamped["unison_voices"])))
        
        # Unison spread: 0-50 cents
        if "unison_spread" in clamped:
            clamped["unison_spread"] = max(0, min(50, clamped["unison_spread"]))
        
        return clamped


def description_to_params(description: str) -> SynthParams:
    """
    Quick function to convert description to SynthParams.
    
    Args:
        description: Natural language description
    
    Returns:
        SynthParams object
    """
    mapper = TimbreMapper()
    result = mapper.map_description(description)
    return result.synth_params


def get_description_params(description: str) -> Dict[str, Any]:
    """
    Get all parameters (synth + effects) for a description.
    
    Args:
        description: Natural language description
    
    Returns:
        Dictionary with all parameters
    """
    mapper = TimbreMapper()
    result = mapper.map_description(description)
    
    return {
        "synth": result.synth_params.to_dict(),
        "effects": result.effects_params,
        "descriptors": result.descriptors_used,
        "confidence": result.confidence,
    }

