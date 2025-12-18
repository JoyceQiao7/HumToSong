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
from scipy import signal

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    SAMPLE_RATE,
    PITCH_CONFIDENCE_THRESHOLD,
    PITCH_METHOD,
    PITCH_MODEL,
    MIN_NOTE_DURATION,
)
from utils.midi_utils import freq_to_midi, quantize_pitch


# =============================================================================
# Audio Preprocessing for Voice Isolation
# =============================================================================

def preprocess_for_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Preprocess audio to isolate human voice for better pitch detection.
    
    Steps:
    1. Bandpass filter for human voice range (80Hz - 1000Hz for humming)
    2. Normalize audio level
    3. Remove silence/noise below threshold
    
    Args:
        audio: Input audio signal
        sr: Sample rate
    
    Returns:
        Preprocessed audio optimized for pitch detection
    """
    # 1. Bandpass filter for human voice/humming frequencies
    # Humming typically ranges from 80Hz to 500Hz (fundamental)
    # We extend to 1000Hz to capture harmonics that help with detection
    low_freq = 80    # Hz - below typical humming range
    high_freq = 1000  # Hz - above typical humming fundamental
    
    # Design bandpass filter
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    # Apply 4th order Butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    audio_filtered = signal.filtfilt(b, a, audio)
    
    # 2. Normalize to consistent level
    max_val = np.max(np.abs(audio_filtered))
    if max_val > 0:
        audio_filtered = audio_filtered / max_val * 0.9
    
    # 3. Simple noise gate - zero out very quiet sections
    noise_threshold = 0.02
    envelope = np.abs(audio_filtered)
    # Smooth the envelope
    window_size = int(sr * 0.05)  # 50ms window
    if window_size > 1:
        envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
    
    # Apply noise gate
    audio_filtered = np.where(envelope > noise_threshold, audio_filtered, 0)
    
    return audio_filtered.astype(np.float32)


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
    method: str = PITCH_METHOD,
    model_capacity: str = PITCH_MODEL,
    step_size: int = 25,
    confidence_threshold: float = PITCH_CONFIDENCE_THRESHOLD,
    preprocess: bool = True,
) -> PitchResult:
    """
    Detect pitch using PYIN (fast) or CREPE (slow but accurate).
    
    For humming/singing, PYIN is recommended (10-20x faster than CREPE).
    
    Args:
        audio: Audio signal (mono, float)
        sr: Sample rate
        method: 'pyin' (fast, recommended) or 'crepe' (slow)
        model_capacity: CREPE model size if using CREPE
        step_size: Hop size in milliseconds (for CREPE)
        confidence_threshold: Minimum confidence for voiced detection
        preprocess: Apply voice isolation preprocessing
    
    Returns:
        PitchResult object with pitch data
    """
    import time as time_module
    
    audio_duration = len(audio) / sr
    print(f"[Pitch] Input: {audio_duration:.2f}s audio at {sr}Hz")
    
    # Preprocess audio to isolate voice
    if preprocess:
        t0 = time_module.time()
        audio = preprocess_for_pitch(audio, sr)
        print(f"[Pitch] Preprocessing took {time_module.time() - t0:.2f}s")
    
    # Check if audio has content after preprocessing
    max_amp = np.max(np.abs(audio))
    print(f"[Pitch] Max amplitude after preprocessing: {max_amp:.4f}")
    
    if max_amp < 0.01:
        print("[Pitch] WARNING: Audio is mostly silent after preprocessing")
        return PitchResult(
            times=np.array([0.0]),
            frequencies=np.array([0.0]),
            confidence=np.array([0.0]),
            midi_pitches=np.array([0.0]),
        )
    
    # Use PYIN by default (much faster for monophonic sources)
    if method == "pyin":
        return detect_pitch_pyin(audio, sr, confidence_threshold, preprocess=False)
    
    # CREPE method (slower but can be more accurate)
    try:
        import crepe
        
        estimated_frames = int(audio_duration * 1000 / step_size)
        print(f"[Pitch] Starting CREPE (model={model_capacity}, step={step_size}ms, ~{estimated_frames} frames)")
        
        t0 = time_module.time()
        
        time, frequency, confidence, _ = crepe.predict(
            audio,
            sr,
            model_capacity=model_capacity,
            step_size=step_size,
            viterbi=False,
        )
        
        elapsed = time_module.time() - t0
        print(f"[Pitch] CREPE completed in {elapsed:.2f}s ({len(time)} frames, {len(time)/elapsed:.0f} frames/sec)")
        
    except ImportError as e:
        print(f"[Pitch] CREPE not available: {e}")
        print("[Pitch] Using PYIN instead...")
        return detect_pitch_pyin(audio, sr, confidence_threshold, preprocess=False)
    except Exception as e:
        print(f"[Pitch] CREPE error: {e}")
        print("[Pitch] Falling back to PYIN...")
        return detect_pitch_pyin(audio, sr, confidence_threshold, preprocess=False)
    
    # Process CREPE results
    frequency = np.where(confidence >= confidence_threshold, frequency, 0)
    
    midi_pitches = np.zeros_like(frequency)
    voiced_mask = frequency > 0
    midi_pitches[voiced_mask] = freq_to_midi(frequency[voiced_mask])
    
    voiced_count = np.sum(voiced_mask)
    print(f"[Pitch] Result: {voiced_count}/{len(time)} frames voiced ({100*voiced_count/len(time):.1f}%)")
    
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
    preprocess: bool = True,
) -> PitchResult:
    """
    Fast pitch detection using librosa's PYIN algorithm.
    
    PYIN is 10-20x faster than CREPE and works excellently for 
    monophonic sources like humming and singing.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        confidence_threshold: Minimum probability threshold
        preprocess: Apply voice isolation preprocessing
    
    Returns:
        PitchResult object
    """
    import time as time_module
    
    # Preprocess for voice isolation
    if preprocess:
        audio = preprocess_for_pitch(audio, sr)
    
    audio_duration = len(audio) / sr
    print(f"[Pitch] Running PYIN on {audio_duration:.2f}s audio...")
    
    t0 = time_module.time()
    
    # Run PYIN with optimized settings for humming/singing
    # Humming range: typically C2 (65Hz) to C5 (523Hz)
    # Larger hop_length = faster processing
    hop_length = 512  # Balance between speed and time resolution
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C5'),  # ~523 Hz
        sr=sr,
        hop_length=hop_length,
        frame_length=2048,  # Good for voice
    )
    
    elapsed = time_module.time() - t0
    
    # Create time array
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    
    # Handle NaN values
    f0 = np.nan_to_num(f0, nan=0.0)
    confidence = voiced_probs if voiced_probs is not None else np.ones_like(f0)
    confidence = np.nan_to_num(confidence, nan=0.0)
    
    # Mark low-confidence as unvoiced
    f0 = np.where(confidence >= confidence_threshold, f0, 0)
    
    # Convert to MIDI
    midi_pitches = np.zeros_like(f0)
    voiced_mask = f0 > 0
    if np.any(voiced_mask):
        midi_pitches[voiced_mask] = freq_to_midi(f0[voiced_mask])
    
    voiced_count = np.sum(voiced_mask)
    print(f"[Pitch] PYIN completed in {elapsed:.2f}s ({len(f0)} frames, {int(len(f0)/elapsed)} frames/sec)")
    print(f"[Pitch] Result: {voiced_count}/{len(f0)} frames voiced ({100*voiced_count/len(f0):.1f}%)")
    
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

