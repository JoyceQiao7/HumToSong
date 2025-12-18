"""
Audio I/O utilities for HumToHarmony.

Provides functions for loading, saving, and basic processing of audio files.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import soundfile as sf
import librosa
import audioread

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, SILENCE_THRESHOLD

# Formats natively supported by soundfile (libsndfile)
SOUNDFILE_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def load_audio(
    file_path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.
    
    Uses soundfile for WAV/FLAC/OGG (native support),
    and audioread (ffmpeg) directly for MP3 and other formats.
    
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
    
    file_ext = file_path.suffix.lower()
    
    # Use soundfile for natively supported formats (faster, no warnings)
    if file_ext in SOUNDFILE_FORMATS:
        audio, loaded_sr = sf.read(str(file_path), dtype='float32')
        
        # Handle stereo to mono conversion
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr is not None and loaded_sr != sr:
            audio = librosa.resample(audio, orig_sr=loaded_sr, target_sr=sr)
            loaded_sr = sr
    else:
        # Use audioread directly for MP3 and other formats (uses ffmpeg)
        audio, loaded_sr = _load_with_audioread(str(file_path), sr, mono)
    
    if normalize:
        audio = normalize_audio(audio)
    
    return audio, loaded_sr


def _load_with_audioread(
    file_path: str,
    target_sr: int = SAMPLE_RATE,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio using audioread (ffmpeg backend) directly.
    
    This avoids the PySoundFile warning that librosa.load() produces.
    """
    with audioread.audio_open(file_path) as f:
        native_sr = f.samplerate
        n_channels = f.channels
        
        # Read all audio data
        audio_data = []
        for buf in f:
            audio_data.append(buf)
        
        # Convert to numpy array
        audio_bytes = b''.join(audio_data)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Normalize int16 to float [-1, 1]
        audio = audio / 32768.0
        
        # Reshape for multi-channel
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            if mono:
                audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if target_sr is not None and native_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=target_sr)
            native_sr = target_sr
        
        return audio, native_sr


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
    # Handle empty or None audio
    if audio is None or len(audio) == 0:
        return audio if audio is not None else np.array([])
    
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
    # Handle empty or very short audio
    if audio is None or len(audio) < frame_length:
        return audio, 0, len(audio) if audio is not None else 0
    
    # Use librosa's trim function
    trimmed, index = librosa.effects.trim(
        audio,
        top_db=20 * np.log10(threshold) if threshold > 0 else -60,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # If trimming resulted in empty audio, return original
    if len(trimmed) == 0:
        return audio, 0, len(audio)
    
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


def prepare_vocal_track(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_duration: float = None,
) -> np.ndarray:
    """
    Prepare the original humming audio as a vocal track.
    
    Applies voice-optimized processing:
    - Bandpass filter for voice clarity
    - Normalization
    - Gentle compression
    - Fade in/out
    
    Args:
        audio: Original humming audio
        sr: Sample rate
        target_duration: Target duration in seconds (pads/crops if needed)
    
    Returns:
        Processed vocal track
    """
    from scipy import signal
    
    # 1. Bandpass filter for voice (wider than pitch detection)
    # Human voice fundamentals: 80Hz - 1100Hz
    # Plus harmonics for naturalness
    low_freq = 80
    high_freq = 5000  # Include harmonics
    
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)
    
    # Apply butterworth bandpass
    b, a = signal.butter(4, [low, high], btype='band')
    vocal = signal.filtfilt(b, a, audio)
    
    # 2. Gentle compression (reduce dynamic range)
    # Simple soft-knee compression
    threshold = 0.3
    ratio = 3.0
    above_threshold = np.abs(vocal) > threshold
    vocal[above_threshold] = np.sign(vocal[above_threshold]) * (
        threshold + (np.abs(vocal[above_threshold]) - threshold) / ratio
    )
    
    # 3. Normalize
    max_val = np.max(np.abs(vocal))
    if max_val > 0:
        vocal = vocal / max_val * 0.85
    
    # 4. Apply fades
    vocal = apply_fade(vocal, fade_in_ms=50, fade_out_ms=200, sr=sr)
    
    # 5. Adjust duration if needed
    if target_duration is not None:
        target_samples = int(target_duration * sr)
        current_samples = len(vocal)
        
        if current_samples < target_samples:
            # Pad with silence
            padding = np.zeros(target_samples - current_samples)
            vocal = np.concatenate([vocal, padding])
        elif current_samples > target_samples:
            # Crop (with fade out)
            vocal = vocal[:target_samples]
            vocal = apply_fade(vocal, fade_in_ms=0, fade_out_ms=200, sr=sr)
    
    return vocal.astype(np.float32)

