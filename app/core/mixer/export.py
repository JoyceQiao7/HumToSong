"""
Audio export functionality for HumToHarmony.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import soundfile as sf

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAMPLE_RATE, PROJECTS_DIR


def export_wav(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    normalize: bool = True,
) -> Path:
    """
    Export audio as WAV file.
    
    Args:
        audio: Audio data (mono or stereo)
        file_path: Output file path
        sr: Sample rate
        normalize: Normalize before export
    
    Returns:
        Path to exported file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .wav extension
    if file_path.suffix.lower() != '.wav':
        file_path = file_path.with_suffix('.wav')
    
    # Prepare audio
    if audio.ndim == 2:
        # Stereo: transpose to (samples, channels)
        audio_out = audio.T
    else:
        audio_out = audio
    
    # Normalize
    if normalize:
        max_val = np.max(np.abs(audio_out))
        if max_val > 0:
            audio_out = audio_out * (0.95 / max_val)
    
    # Ensure float32
    audio_out = audio_out.astype(np.float32)
    
    # Write
    sf.write(str(file_path), audio_out, sr)
    
    return file_path


def export_mp3(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    bitrate: str = "192k",
    normalize: bool = True,
) -> Optional[Path]:
    """
    Export audio as MP3 file.
    
    Requires pydub and ffmpeg to be installed.
    
    Args:
        audio: Audio data
        file_path: Output file path
        sr: Sample rate
        bitrate: MP3 bitrate
        normalize: Normalize before export
    
    Returns:
        Path to exported file, or None if export failed
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        print("pydub not installed. Falling back to WAV export.")
        return export_wav(audio, Path(file_path).with_suffix('.wav'), sr, normalize)
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .mp3 extension
    if file_path.suffix.lower() != '.mp3':
        file_path = file_path.with_suffix('.mp3')
    
    # First export as WAV temporarily
    temp_wav = file_path.with_suffix('.temp.wav')
    export_wav(audio, temp_wav, sr, normalize)
    
    try:
        # Convert to MP3
        sound = AudioSegment.from_wav(str(temp_wav))
        sound.export(str(file_path), format="mp3", bitrate=bitrate)
        
        # Clean up temp file
        temp_wav.unlink()
        
        return file_path
    
    except Exception as e:
        print(f"MP3 export failed: {e}")
        # Return the WAV file instead
        if temp_wav.exists():
            final_wav = file_path.with_suffix('.wav')
            temp_wav.rename(final_wav)
            return final_wav
        return None


def export_midi(
    notes: List,
    file_path: Union[str, Path],
    tempo: float = 120.0,
    track_name: str = "Melody",
) -> Path:
    """
    Export notes as MIDI file.
    
    Args:
        notes: List of Note objects (with times in beats)
        file_path: Output file path
        tempo: Tempo in BPM
        track_name: Name for the MIDI track
    
    Returns:
        Path to exported file
    """
    from midiutil import MIDIFile
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .mid extension
    if file_path.suffix.lower() not in ['.mid', '.midi']:
        file_path = file_path.with_suffix('.mid')
    
    # Create MIDI file
    midi = MIDIFile(1)  # One track
    
    track = 0
    channel = 0
    time = 0
    
    midi.addTrackName(track, time, track_name)
    midi.addTempo(track, time, tempo)
    
    # Add notes
    for note in notes:
        midi.addNote(
            track=track,
            channel=channel,
            pitch=note.pitch,
            time=note.start_time,
            duration=note.duration,
            volume=note.velocity,
        )
    
    # Write file
    with open(file_path, 'wb') as f:
        midi.writeFile(f)
    
    return file_path


def export_project(
    melody_audio: np.ndarray,
    harmony_audio: np.ndarray,
    bass_audio: np.ndarray,
    mixed_audio: np.ndarray,
    melody_notes: List,
    project_name: str,
    tempo: float = 120.0,
    sr: int = SAMPLE_RATE,
) -> dict:
    """
    Export a complete project (all stems and MIDI).
    
    Args:
        melody_audio: Melody track audio
        harmony_audio: Harmony track audio
        bass_audio: Bass track audio
        mixed_audio: Full mix audio
        melody_notes: Melody notes for MIDI export
        project_name: Name of the project
        tempo: Tempo in BPM
        sr: Sample rate
    
    Returns:
        Dictionary of exported file paths
    """
    # Create project folder
    project_dir = PROJECTS_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    exports = {}
    
    # Export individual stems
    if len(melody_audio) > 0:
        exports['melody_wav'] = export_wav(
            melody_audio, project_dir / "melody.wav", sr
        )
    
    if len(harmony_audio) > 0:
        exports['harmony_wav'] = export_wav(
            harmony_audio, project_dir / "harmony.wav", sr
        )
    
    if len(bass_audio) > 0:
        exports['bass_wav'] = export_wav(
            bass_audio, project_dir / "bass.wav", sr
        )
    
    # Export full mix
    if len(mixed_audio) > 0:
        exports['mix_wav'] = export_wav(
            mixed_audio, project_dir / "mix.wav", sr
        )
        exports['mix_mp3'] = export_mp3(
            mixed_audio, project_dir / "mix.mp3", sr
        )
    
    # Export MIDI
    if melody_notes:
        exports['midi'] = export_midi(
            melody_notes, project_dir / "melody.mid", tempo
        )
    
    return exports


def get_audio_bytes(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    format: str = "wav",
) -> bytes:
    """
    Get audio as bytes (for streaming/preview).
    
    Args:
        audio: Audio data
        sr: Sample rate
        format: Output format ('wav')
    
    Returns:
        Audio data as bytes
    """
    import io
    
    # Prepare audio
    if audio.ndim == 2:
        audio_out = audio.T
    else:
        audio_out = audio
    
    # Normalize
    max_val = np.max(np.abs(audio_out))
    if max_val > 0:
        audio_out = audio_out * (0.95 / max_val)
    
    audio_out = audio_out.astype(np.float32)
    
    # Write to bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_out, sr, format=format)
    buffer.seek(0)
    
    return buffer.read()

