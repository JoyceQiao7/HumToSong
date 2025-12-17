"""
Audio I/O utilities for HumToHarmony.

Provides functions for loading, saving, and basic processing of audio files.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import soundfile as sf
import librosa

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, SILENCE_THRESHOLD


def load_audio(
    file_path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.
    
    Args:
        file_path: Path to the audio file
        sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
        normalize: Normalize audio to [-1, 1] if True
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load with librosa for format flexibility
    audio, loaded_sr = librosa.load(
        str(file_path),
        sr=sr,
        mono=mono
    )
    
    if normalize:
        audio = normalize_audio(audio)
    
    return audio, loaded_sr


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    normalize: bool = True,
    format: str = None
) -> Path:
    """
    Save audio data to a file.
    
    Args:
        audio: Audio data as numpy array
        file_path: Output file path
        sr: Sample rate
        normalize: Normalize before saving
        format: Output format (inferred from extension if None)
    
    Returns:
        Path to saved file
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize if requested
    if normalize:
        audio = normalize_audio(audio)
    
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Save using soundfile
    sf.write(str(file_path), audio, sr)
    
    return file_path


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to a target peak level.
    
    Args:
        audio: Input audio array
        target_peak: Target peak amplitude (0-1)
    
    Returns:
        Normalized audio array
    """
    peak = np.max(np.abs(audio))
    
    if peak > 0:
        return audio * (target_peak / peak)
    
    return audio


def trim_silence(
    audio: np.ndarray,
    threshold: float = SILENCE_THRESHOLD,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, int, int]:
    """
    Trim silence from the beginning and end of audio.
    
    Args:
        audio: Input audio array
        threshold: Silence threshold (0-1)
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
    
    Returns:
        Tuple of (trimmed_audio, start_sample, end_sample)
    """
    # Use librosa's trim function
    trimmed, index = librosa.effects.trim(
        audio,
        top_db=20 * np.log10(threshold) if threshold > 0 else -60,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return trimmed, index[0], index[1]


def get_duration(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Get the duration of audio in seconds.
    
    Args:
        audio: Audio array
        sr: Sample rate
    
    Returns:
        Duration in seconds
    """
    return len(audio) / sr


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Resample audio to a different sample rate.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def audio_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono.
    
    Args:
        audio: Input audio (can be mono or stereo)
    
    Returns:
        Mono audio array
    """
    if audio.ndim == 1:
        return audio
    
    return np.mean(audio, axis=0)


def mix_audio(
    tracks: list,
    volumes: list = None,
    normalize_output: bool = True
) -> np.ndarray:
    """
    Mix multiple audio tracks together.
    
    Args:
        tracks: List of audio arrays (must be same length)
        volumes: Optional list of volume multipliers (0-1)
        normalize_output: Normalize the mixed output
    
    Returns:
        Mixed audio array
    """
    if not tracks:
        raise ValueError("No tracks provided for mixing")
    
    # Get the maximum length
    max_length = max(len(t) for t in tracks)
    
    # Set default volumes
    if volumes is None:
        volumes = [1.0] * len(tracks)
    
    # Pad tracks to same length and mix
    mixed = np.zeros(max_length)
    
    for track, volume in zip(tracks, volumes):
        padded = np.zeros(max_length)
        padded[:len(track)] = track
        mixed += padded * volume
    
    if normalize_output:
        mixed = normalize_audio(mixed)
    
    return mixed


def apply_fade(
    audio: np.ndarray,
    fade_in_ms: float = 10,
    fade_out_ms: float = 10,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Apply fade in and fade out to audio.
    
    Args:
        audio: Input audio array
        fade_in_ms: Fade in duration in milliseconds
        fade_out_ms: Fade out duration in milliseconds
        sr: Sample rate
    
    Returns:
        Audio with fades applied
    """
    audio = audio.copy()
    
    # Calculate fade lengths in samples
    fade_in_samples = int(fade_in_ms * sr / 1000)
    fade_out_samples = int(fade_out_ms * sr / 1000)
    
    # Apply fade in
    if fade_in_samples > 0 and fade_in_samples < len(audio):
        fade_in = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in
    
    # Apply fade out
    if fade_out_samples > 0 and fade_out_samples < len(audio):
        fade_out = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out
    
    return audio

