"""
Musical key detection for HumToHarmony.

Analyzes pitch content to determine the most likely key/scale.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.midi_utils import NOTE_NAMES


@dataclass
class KeyResult:
    """Result of key detection."""
    key: str                    # Root note (e.g., "C", "F#")
    scale: str                  # Scale type ("major" or "minor")
    confidence: float           # Confidence score (0-1)
    correlation: float          # Correlation with key profile
    alternatives: List[Tuple[str, str, float]]  # Alternative key/scale options
    
    def __str__(self) -> str:
        return f"{self.key} {self.scale}"
    
    @property
    def full_name(self) -> str:
        """Get full key name like 'C major' or 'A minor'."""
        return f"{self.key} {self.scale}"


# Krumhansl-Schmuckler key profiles
# These represent the perceptual "weight" of each pitch class in major/minor keys
MAJOR_PROFILE = np.array([
    6.35,  # C
    2.23,  # C#
    3.48,  # D
    2.33,  # D#
    4.38,  # E
    4.09,  # F
    2.52,  # F#
    5.19,  # G
    2.39,  # G#
    3.66,  # A
    2.29,  # A#
    2.88,  # B
])

MINOR_PROFILE = np.array([
    6.33,  # C
    2.68,  # C#
    3.52,  # D
    5.38,  # D#/Eb
    2.60,  # E
    3.53,  # F
    2.54,  # F#
    4.75,  # G
    3.98,  # G#/Ab
    2.69,  # A
    3.34,  # A#/Bb
    3.17,  # B
])


def detect_key(
    notes: List,
    method: str = "krumhansl",
) -> KeyResult:
    """
    Detect the musical key from a list of notes.
    
    Args:
        notes: List of Note objects with 'pitch' attribute
        method: Detection method ("krumhansl" or "simple")
    
    Returns:
        KeyResult object with detected key
    """
    if not notes:
        return KeyResult(
            key="C",
            scale="major",
            confidence=0.0,
            correlation=0.0,
            alternatives=[],
        )
    
    # Build pitch class histogram
    pitch_histogram = _build_pitch_histogram(notes)
    
    if method == "krumhansl":
        return _detect_key_krumhansl(pitch_histogram)
    else:
        return _detect_key_simple(pitch_histogram)


def _build_pitch_histogram(notes: List, weighted: bool = True) -> np.ndarray:
    """
    Build a histogram of pitch classes from notes.
    
    Args:
        notes: List of Note objects
        weighted: Weight by note duration if True
    
    Returns:
        Array of 12 values (one per pitch class)
    """
    histogram = np.zeros(12)
    
    for note in notes:
        pitch_class = note.pitch % 12
        
        if weighted:
            # Weight by duration
            weight = note.duration
        else:
            weight = 1.0
        
        histogram[pitch_class] += weight
    
    # Normalize
    total = np.sum(histogram)
    if total > 0:
        histogram = histogram / total
    
    return histogram


def _detect_key_krumhansl(pitch_histogram: np.ndarray) -> KeyResult:
    """
    Detect key using Krumhansl-Schmuckler algorithm.
    
    Correlates the pitch histogram with key profiles for all 24 major/minor keys.
    """
    correlations = []
    
    # Test all 24 keys
    for root in range(12):
        # Rotate profiles to match root
        major_rotated = np.roll(MAJOR_PROFILE, root)
        minor_rotated = np.roll(MINOR_PROFILE, root)
        
        # Calculate correlation
        major_corr = np.corrcoef(pitch_histogram, major_rotated)[0, 1]
        minor_corr = np.corrcoef(pitch_histogram, minor_rotated)[0, 1]
        
        # Handle NaN (can happen with constant histogram)
        if np.isnan(major_corr):
            major_corr = 0.0
        if np.isnan(minor_corr):
            minor_corr = 0.0
        
        correlations.append((root, "major", major_corr))
        correlations.append((root, "minor", minor_corr))
    
    # Sort by correlation (descending)
    correlations.sort(key=lambda x: x[2], reverse=True)
    
    # Best match
    best_root, best_scale, best_corr = correlations[0]
    
    # Calculate confidence based on difference from second best
    second_corr = correlations[1][2]
    confidence = (best_corr - second_corr) / max(0.001, abs(best_corr))
    confidence = min(1.0, max(0.0, confidence))
    
    # Get top alternatives
    alternatives = [
        (NOTE_NAMES[root], scale, corr)
        for root, scale, corr in correlations[1:4]
    ]
    
    return KeyResult(
        key=NOTE_NAMES[best_root],
        scale=best_scale,
        confidence=confidence,
        correlation=best_corr,
        alternatives=alternatives,
    )


def _detect_key_simple(pitch_histogram: np.ndarray) -> KeyResult:
    """
    Simple key detection based on most common pitch classes.
    
    Less accurate than Krumhansl but faster.
    """
    # Find most common pitch classes
    top_pcs = np.argsort(pitch_histogram)[::-1][:4]
    
    # Check if pitches fit major or minor scale patterns
    best_match = None
    best_score = 0
    
    for root in range(12):
        # Major scale pitch classes
        major_pcs = set([(root + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]])
        # Minor scale pitch classes
        minor_pcs = set([(root + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]])
        
        # Count matches with top pitch classes
        top_set = set(top_pcs)
        major_score = len(top_set & major_pcs)
        minor_score = len(top_set & minor_pcs)
        
        if major_score > best_score:
            best_score = major_score
            best_match = (root, "major")
        if minor_score > best_score:
            best_score = minor_score
            best_match = (root, "minor")
    
    if best_match is None:
        best_match = (0, "major")
    
    confidence = best_score / 4.0  # 4 top pitch classes
    
    return KeyResult(
        key=NOTE_NAMES[best_match[0]],
        scale=best_match[1],
        confidence=confidence,
        correlation=0.0,
        alternatives=[],
    )


def get_scale_notes(key: str, scale: str = "major") -> List[str]:
    """
    Get note names in a scale.
    
    Args:
        key: Root note (e.g., "C", "F#")
        scale: Scale type
    
    Returns:
        List of note names in the scale
    """
    # Scale intervals in semitones
    INTERVALS = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
    }
    
    if scale not in INTERVALS:
        scale = "major"
    
    # Get root pitch class
    root_pc = NOTE_NAMES.index(key.upper()) if key.upper() in NOTE_NAMES else 0
    
    # Build scale
    scale_notes = []
    for interval in INTERVALS[scale]:
        pc = (root_pc + interval) % 12
        scale_notes.append(NOTE_NAMES[pc])
    
    return scale_notes


def get_relative_key(key: str, scale: str) -> Tuple[str, str]:
    """
    Get the relative major/minor key.
    
    Args:
        key: Root note
        scale: Scale type
    
    Returns:
        Tuple of (relative_key, relative_scale)
    """
    root_pc = NOTE_NAMES.index(key.upper()) if key.upper() in NOTE_NAMES else 0
    
    if scale == "major":
        # Relative minor is 3 semitones down
        relative_pc = (root_pc - 3) % 12
        return NOTE_NAMES[relative_pc], "minor"
    else:
        # Relative major is 3 semitones up
        relative_pc = (root_pc + 3) % 12
        return NOTE_NAMES[relative_pc], "major"


def get_parallel_key(key: str, scale: str) -> Tuple[str, str]:
    """
    Get the parallel major/minor key (same root, different scale).
    
    Args:
        key: Root note
        scale: Scale type
    
    Returns:
        Tuple of (same_key, opposite_scale)
    """
    if scale == "major":
        return key, "minor"
    else:
        return key, "major"

