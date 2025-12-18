"""
Note and rhythm quantization for HumToHarmony.

Quantizes detected notes to:
1. Musical grid (1/4, 1/8, 1/16 notes)
2. Scale degrees (snap to key)
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAMPLE_RATE, DEFAULT_TEMPO, QUANTIZE_RESOLUTION, MIN_NOTE_DURATION
from utils.midi_utils import get_scale_pitches, snap_to_scale


def quantize_notes(
    notes: List,
    tempo: float = DEFAULT_TEMPO,
    resolution: int = QUANTIZE_RESOLUTION,
    key: str = "C",
    scale: str = "major",
    snap_to_grid: bool = True,
    snap_pitch: bool = True,
    add_musicality: bool = True,
) -> List:
    """
    Quantize notes to musical grid and scale with musicality enhancements.
    
    Args:
        notes: List of Note objects (times in seconds or beats)
        tempo: Tempo in BPM
        resolution: Quantization resolution (16 = 16th notes)
        key: Key for pitch snapping (e.g., "C", "F#")
        scale: Scale type (e.g., "major", "minor")
        snap_to_grid: Whether to quantize timing
        snap_pitch: Whether to snap pitches to scale
        add_musicality: Add dynamics, legato, and phrasing
    
    Returns:
        List of quantized Note objects
    """
    if not notes:
        return []
    
    # Get scale pitches for snapping
    scale_pitches = get_scale_pitches(key, scale) if snap_pitch else None
    
    # Grid size in beats
    grid_size = 4.0 / resolution
    
    quantized = []
    
    for i, note in enumerate(notes):
        new_pitch = note.pitch
        new_start = note.start_time
        new_duration = note.duration
        new_velocity = note.velocity
        
        # Snap pitch to scale
        if snap_pitch and scale_pitches:
            new_pitch = snap_to_scale(note.pitch, scale_pitches)
        
        # Quantize timing
        if snap_to_grid:
            # Snap start time to grid
            new_start = round(note.start_time / grid_size) * grid_size
            
            # Quantize duration to grid (minimum 1 grid unit)
            new_duration = max(grid_size, round(note.duration / grid_size) * grid_size)
        
        # Add musicality
        if add_musicality and i > 0:
            prev_note = quantized[-1]
            
            # Add legato (slight overlap) for smooth melodies
            gap = new_start - (prev_note.start_time + prev_note.duration)
            if gap < grid_size * 0.5 and abs(new_pitch - prev_note.pitch) <= 2:
                # Extend previous note slightly for legato
                prev_note.duration += min(gap * 0.5, grid_size * 0.25)
            
            # Add dynamics variation
            # Longer notes get slightly softer, shorter notes punchier
            if new_duration > 2.0:
                new_velocity = int(new_velocity * 0.9)  # Longer = softer
            elif new_duration < 0.5:
                new_velocity = int(min(127, new_velocity * 1.1))  # Shorter = punchier
            
            # Emphasize downbeats (notes on beat 1 of measure)
            if abs(new_start % 4.0) < grid_size * 0.1:
                new_velocity = int(min(127, new_velocity * 1.15))
        
        # Create new note with quantized values
        from .detector import Note
        quantized.append(Note(
            pitch=int(new_pitch),
            start_time=new_start,
            duration=new_duration,
            velocity=max(40, min(127, new_velocity)),  # Clamp velocity
            confidence=note.confidence,
        ))
    
    return quantized


def detect_tempo(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    start_bpm: float = 120.0,
) -> Tuple[float, np.ndarray]:
    """
    Detect tempo from audio using librosa.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        start_bpm: Prior tempo estimate
    
    Returns:
        Tuple of (tempo_bpm, beat_times)
    """
    # Estimate tempo
    tempo, beat_frames = librosa.beat.beat_track(
        y=audio,
        sr=sr,
        start_bpm=start_bpm,
        units='frames',
    )
    
    # Convert beat frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # tempo might be an array in newer librosa versions
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else start_bpm
    
    return float(tempo), beat_times


def estimate_tempo_from_notes(
    notes: List,
    min_tempo: float = 60.0,
    max_tempo: float = 200.0,
) -> float:
    """
    Estimate tempo from note inter-onset intervals.
    
    Args:
        notes: List of Note objects
        min_tempo: Minimum reasonable tempo
        max_tempo: Maximum reasonable tempo
    
    Returns:
        Estimated tempo in BPM
    """
    if len(notes) < 2:
        return DEFAULT_TEMPO
    
    # Calculate inter-onset intervals
    onsets = sorted([n.start_time for n in notes])
    intervals = np.diff(onsets)
    
    # Filter out very short or very long intervals
    intervals = intervals[intervals > 0.1]  # > 100ms
    intervals = intervals[intervals < 2.0]  # < 2 seconds
    
    if len(intervals) == 0:
        return DEFAULT_TEMPO
    
    # Find most common interval (likely beat or half-beat)
    median_interval = np.median(intervals)
    
    # Convert to BPM (assuming interval is a beat)
    tempo = 60.0 / median_interval
    
    # Adjust if tempo is out of range
    while tempo < min_tempo:
        tempo *= 2
    while tempo > max_tempo:
        tempo /= 2
    
    return tempo


def align_notes_to_beats(
    notes: List,
    beat_times: np.ndarray,
    tolerance: float = 0.1,
) -> List:
    """
    Align note start times to detected beat times.
    
    Args:
        notes: List of Note objects
        beat_times: Array of beat times in seconds
        tolerance: Maximum time deviation to snap (seconds)
    
    Returns:
        List of aligned Note objects
    """
    if len(beat_times) == 0:
        return notes
    
    aligned = []
    
    for note in notes:
        # Find nearest beat
        distances = np.abs(beat_times - note.start_time)
        nearest_idx = np.argmin(distances)
        nearest_beat = beat_times[nearest_idx]
        
        # Snap if within tolerance
        if distances[nearest_idx] <= tolerance:
            new_start = nearest_beat
        else:
            new_start = note.start_time
        
        from .detector import Note
        aligned.append(Note(
            pitch=note.pitch,
            start_time=new_start,
            duration=note.duration,
            velocity=note.velocity,
            confidence=note.confidence,
        ))
    
    return aligned


def merge_short_notes(
    notes: List,
    min_duration: float = MIN_NOTE_DURATION,
) -> List:
    """
    Merge very short notes with adjacent notes of similar pitch.
    
    Args:
        notes: List of Note objects (sorted by start time)
        min_duration: Minimum note duration in beats
    
    Returns:
        List of merged Note objects
    """
    if len(notes) <= 1:
        return notes
    
    # Sort by start time
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    merged = []
    current = sorted_notes[0]
    
    for next_note in sorted_notes[1:]:
        # Check if notes should be merged
        same_pitch = current.pitch == next_note.pitch
        short_gap = (next_note.start_time - (current.start_time + current.duration)) < min_duration
        current_short = current.duration < min_duration
        
        if same_pitch and (short_gap or current_short):
            # Merge: extend current note
            new_end = next_note.start_time + next_note.duration
            from .detector import Note
            current = Note(
                pitch=current.pitch,
                start_time=current.start_time,
                duration=new_end - current.start_time,
                velocity=max(current.velocity, next_note.velocity),
                confidence=(current.confidence + next_note.confidence) / 2,
            )
        else:
            # Keep current, move to next
            if current.duration >= min_duration:
                merged.append(current)
            current = next_note
    
    # Don't forget the last note
    if current.duration >= min_duration:
        merged.append(current)
    
    return merged


def remove_duplicate_notes(
    notes: List,
    time_tolerance: float = 0.05,
) -> List:
    """
    Remove duplicate notes (same pitch at nearly same time).
    
    Args:
        notes: List of Note objects
        time_tolerance: Maximum time difference to consider duplicate
    
    Returns:
        List with duplicates removed
    """
    if len(notes) <= 1:
        return notes
    
    # Sort by start time
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    unique = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        last = unique[-1]
        
        # Check if duplicate
        same_pitch = note.pitch == last.pitch
        same_time = abs(note.start_time - last.start_time) < time_tolerance
        
        if not (same_pitch and same_time):
            unique.append(note)
        else:
            # Keep the one with higher confidence
            if note.confidence > last.confidence:
                unique[-1] = note
    
    return unique

