"""
Pitch detection using CREPE neural network.

CREPE (Convolutional Representation for Pitch Estimation) is a 
state-of-the-art monophonic pitch tracker that works excellently
for humming and singing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    SAMPLE_RATE,
    PITCH_CONFIDENCE_THRESHOLD,
    PITCH_MODEL,
    MIN_NOTE_DURATION,
)
from utils.midi_utils import freq_to_midi, quantize_pitch


@dataclass
class PitchResult:
    """Result of pitch detection."""
    times: np.ndarray           # Time stamps in seconds
    frequencies: np.ndarray     # Frequencies in Hz (0 = unvoiced)
    confidence: np.ndarray      # Confidence values (0-1)
    midi_pitches: np.ndarray    # MIDI pitch numbers (fractional)
    
    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.times[-1] if len(self.times) > 0 else 0.0
    
    def get_voiced_segments(self, threshold: float = PITCH_CONFIDENCE_THRESHOLD) -> List[Tuple[float, float]]:
        """Get list of (start, end) times for voiced segments."""
        voiced = self.confidence >= threshold
        segments = []
        
        in_segment = False
        start = 0.0
        
        for i, (t, v) in enumerate(zip(self.times, voiced)):
            if v and not in_segment:
                start = t
                in_segment = True
            elif not v and in_segment:
                segments.append((start, self.times[i-1]))
                in_segment = False
        
        if in_segment:
            segments.append((start, self.times[-1]))
        
        return segments


@dataclass
class Note:
    """A detected musical note."""
    pitch: int              # MIDI note number
    start_time: float       # Start time in seconds
    duration: float         # Duration in seconds
    velocity: int = 100     # MIDI velocity (0-127)
    confidence: float = 1.0 # Average confidence
    
    def to_dict(self) -> dict:
        return {
            'pitch': self.pitch,
            'start_time': self.start_time,
            'duration': self.duration,
            'velocity': self.velocity,
            'confidence': self.confidence,
        }


def detect_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    model_capacity: str = PITCH_MODEL,
    step_size: int = 10,
    confidence_threshold: float = PITCH_CONFIDENCE_THRESHOLD,
) -> PitchResult:
    """
    Detect pitch using CREPE neural network.
    
    Args:
        audio: Audio signal (mono, float)
        sr: Sample rate
        model_capacity: CREPE model size ('tiny', 'small', 'medium', 'large', 'full')
        step_size: Hop size in milliseconds
        confidence_threshold: Minimum confidence for voiced detection
    
    Returns:
        PitchResult object with pitch data
    """
    try:
        import crepe
        
        # Run CREPE pitch detection
        time, frequency, confidence, _ = crepe.predict(
            audio,
            sr,
            model_capacity=model_capacity,
            step_size=step_size,
            viterbi=True,  # Use Viterbi decoding for smoother pitch
        )
        
    except ImportError:
        # Fallback to librosa's pyin if CREPE not available
        print("CREPE not available, falling back to PYIN")
        return detect_pitch_pyin(audio, sr, confidence_threshold)
    
    # Mark low-confidence frames as unvoiced
    frequency = np.where(confidence >= confidence_threshold, frequency, 0)
    
    # Convert frequencies to MIDI (fractional)
    midi_pitches = np.zeros_like(frequency)
    voiced_mask = frequency > 0
    midi_pitches[voiced_mask] = freq_to_midi(frequency[voiced_mask])
    
    return PitchResult(
        times=time,
        frequencies=frequency,
        confidence=confidence,
        midi_pitches=midi_pitches,
    )


def detect_pitch_pyin(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    confidence_threshold: float = 0.5,
) -> PitchResult:
    """
    Fallback pitch detection using librosa's PYIN algorithm.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        confidence_threshold: Minimum probability threshold
    
    Returns:
        PitchResult object
    """
    # Run PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
    )
    
    # Create time array
    hop_length = 512
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    
    # Handle NaN values
    f0 = np.nan_to_num(f0, nan=0.0)
    
    # Use voiced probability as confidence
    confidence = voiced_probs if voiced_probs is not None else np.ones_like(f0)
    
    # Mark low-confidence as unvoiced
    f0 = np.where(confidence >= confidence_threshold, f0, 0)
    
    # Convert to MIDI
    midi_pitches = np.zeros_like(f0)
    voiced_mask = f0 > 0
    midi_pitches[voiced_mask] = freq_to_midi(f0[voiced_mask])
    
    return PitchResult(
        times=times,
        frequencies=f0,
        confidence=confidence,
        midi_pitches=midi_pitches,
    )


def extract_notes(
    pitch_result: PitchResult,
    min_duration: float = 0.05,
    pitch_tolerance: float = 0.5,
    confidence_threshold: float = PITCH_CONFIDENCE_THRESHOLD,
) -> List[Note]:
    """
    Convert pitch contour to discrete notes.
    
    Uses a segmentation approach:
    1. Find voiced regions
    2. Segment by pitch stability
    3. Create notes from stable segments
    
    Args:
        pitch_result: PitchResult from pitch detection
        min_duration: Minimum note duration in seconds
        pitch_tolerance: Maximum pitch change (in semitones) within a note
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        List of Note objects
    """
    notes = []
    
    # Get voiced frames
    voiced_mask = (pitch_result.confidence >= confidence_threshold) & (pitch_result.frequencies > 0)
    
    if not np.any(voiced_mask):
        return notes
    
    # Find continuous voiced segments
    segments = _find_voiced_segments(voiced_mask)
    
    for start_idx, end_idx in segments:
        # Get pitch values for this segment
        segment_times = pitch_result.times[start_idx:end_idx]
        segment_pitches = pitch_result.midi_pitches[start_idx:end_idx]
        segment_confidence = pitch_result.confidence[start_idx:end_idx]
        
        # Sub-segment by pitch stability
        sub_notes = _segment_by_pitch(
            segment_times,
            segment_pitches,
            segment_confidence,
            pitch_tolerance,
            min_duration,
        )
        
        notes.extend(sub_notes)
    
    return notes


def _find_voiced_segments(voiced_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find continuous voiced segments as (start_idx, end_idx) tuples."""
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, voiced in enumerate(voiced_mask):
        if voiced and not in_segment:
            start_idx = i
            in_segment = True
        elif not voiced and in_segment:
            segments.append((start_idx, i))
            in_segment = False
    
    if in_segment:
        segments.append((start_idx, len(voiced_mask)))
    
    return segments


def _segment_by_pitch(
    times: np.ndarray,
    pitches: np.ndarray,
    confidence: np.ndarray,
    tolerance: float,
    min_duration: float,
) -> List[Note]:
    """
    Segment a voiced region into notes based on pitch stability.
    """
    if len(times) < 2:
        return []
    
    notes = []
    note_start_idx = 0
    
    for i in range(1, len(pitches)):
        pitch_change = abs(pitches[i] - pitches[i-1])
        
        # Check if pitch changed beyond tolerance
        if pitch_change > tolerance:
            # End current note, start new one
            note = _create_note(
                times[note_start_idx:i],
                pitches[note_start_idx:i],
                confidence[note_start_idx:i],
                min_duration,
            )
            if note:
                notes.append(note)
            note_start_idx = i
    
    # Create final note
    note = _create_note(
        times[note_start_idx:],
        pitches[note_start_idx:],
        confidence[note_start_idx:],
        min_duration,
    )
    if note:
        notes.append(note)
    
    return notes


def _create_note(
    times: np.ndarray,
    pitches: np.ndarray,
    confidence: np.ndarray,
    min_duration: float,
) -> Optional[Note]:
    """Create a Note object from a segment."""
    if len(times) < 2:
        return None
    
    duration = times[-1] - times[0]
    
    if duration < min_duration:
        return None
    
    # Use median pitch (more robust to outliers)
    median_pitch = np.median(pitches)
    midi_note = quantize_pitch(median_pitch)
    
    # Calculate average confidence
    avg_confidence = np.mean(confidence)
    
    # Map confidence to velocity (60-120 range)
    velocity = int(60 + avg_confidence * 60)
    velocity = min(127, max(1, velocity))
    
    return Note(
        pitch=midi_note,
        start_time=times[0],
        duration=duration,
        velocity=velocity,
        confidence=avg_confidence,
    )


def notes_to_beats(
    notes: List[Note],
    tempo: float = 120.0,
) -> List[Note]:
    """
    Convert note times from seconds to beats.
    
    Args:
        notes: List of notes with times in seconds
        tempo: Tempo in BPM
    
    Returns:
        List of notes with times in beats
    """
    beat_notes = []
    
    for note in notes:
        beat_note = Note(
            pitch=note.pitch,
            start_time=note.start_time * tempo / 60.0,
            duration=note.duration * tempo / 60.0,
            velocity=note.velocity,
            confidence=note.confidence,
        )
        beat_notes.append(beat_note)
    
    return beat_notes

