"""
Visualization utilities for HumToHarmony.

Provides functions for creating audio visualizations using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, UI_PRIMARY_COLOR, UI_SECONDARY_COLOR, UI_BACKGROUND_COLOR


def plot_waveform(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    title: str = "Waveform",
    color: str = UI_PRIMARY_COLOR,
    height: int = 200
) -> go.Figure:
    """
    Create a waveform visualization.
    
    Args:
        audio: Audio data array
        sr: Sample rate
        title: Plot title
        color: Waveform color
        height: Plot height in pixels
    
    Returns:
        Plotly figure object
    """
    # Downsample for visualization (max 10000 points)
    if len(audio) > 10000:
        factor = len(audio) // 10000
        audio_display = audio[::factor]
    else:
        audio_display = audio
    
    time = np.linspace(0, len(audio) / sr, len(audio_display))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=audio_display,
        mode='lines',
        line=dict(color=color, width=1),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.3])}',
        name='Waveform'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[-1, 1]),
    )
    
    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    title: str = "Spectrogram",
    height: int = 300,
    n_fft: int = 2048,
    hop_length: int = 512
) -> go.Figure:
    """
    Create a spectrogram visualization.
    
    Args:
        audio: Audio data array
        sr: Sample rate
        title: Plot title
        height: Plot height in pixels
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        Plotly figure object
    """
    # Compute spectrogram
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Create time and frequency axes
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Limit frequency range to 0-8kHz for better visualization
    freq_mask = freqs <= 8000
    S_db = S_db[freq_mask, :]
    freqs = freqs[freq_mask]
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=times,
        y=freqs,
        z=S_db,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='dB')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def plot_piano_roll(
    notes: List[dict],
    duration: float = None,
    title: str = "Piano Roll",
    height: int = 400,
    note_range: Tuple[int, int] = (48, 84),  # C3 to C6
    color: str = UI_PRIMARY_COLOR
) -> go.Figure:
    """
    Create a piano roll visualization.
    
    Args:
        notes: List of note dicts with 'pitch', 'start_time', 'duration'
        duration: Total duration in beats (auto-calculated if None)
        title: Plot title
        height: Plot height in pixels
        note_range: (min_pitch, max_pitch) for display
        color: Note color
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if not notes:
        # Empty piano roll
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers'))
    else:
        # Calculate duration if not provided
        if duration is None:
            duration = max(n['start_time'] + n['duration'] for n in notes) + 1
        
        # Draw each note as a rectangle
        for note in notes:
            pitch = note['pitch']
            start = note['start_time']
            dur = note['duration']
            velocity = note.get('velocity', 100)
            
            # Skip notes outside range
            if pitch < note_range[0] or pitch > note_range[1]:
                continue
            
            # Calculate opacity based on velocity
            opacity = 0.4 + (velocity / 127) * 0.6
            
            # Add rectangle for note
            fig.add_shape(
                type="rect",
                x0=start,
                x1=start + dur,
                y0=pitch - 0.4,
                y1=pitch + 0.4,
                fillcolor=color,
                opacity=opacity,
                line=dict(color=color, width=1)
            )
    
    # Add note name labels on y-axis
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    tickvals = list(range(note_range[0], note_range[1] + 1))
    ticktext = [f"{note_names[p % 12]}{p // 12 - 1}" if p % 12 == 0 else "" 
                for p in tickvals]
    
    # Add grid lines for octaves
    for pitch in range(note_range[0], note_range[1] + 1):
        if pitch % 12 == 0:  # C notes
            fig.add_hline(
                y=pitch,
                line=dict(color='rgba(255,255,255,0.3)', width=1),
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (beats)",
        yaxis_title="Pitch",
        height=height,
        margin=dict(l=60, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            dtick=1,
            range=[0, duration] if notes else [0, 8]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            range=[note_range[0] - 1, note_range[1] + 1]
        ),
    )
    
    return fig


def plot_pitch_contour(
    times: np.ndarray,
    pitches: np.ndarray,
    confidence: np.ndarray = None,
    title: str = "Pitch Contour",
    height: int = 300,
    color: str = UI_SECONDARY_COLOR
) -> go.Figure:
    """
    Create a pitch contour visualization.
    
    Args:
        times: Time array in seconds
        pitches: Pitch array in Hz
        confidence: Optional confidence values (0-1)
        title: Plot title
        height: Plot height in pixels
        color: Line color
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Mask out unvoiced regions (pitch = 0)
    voiced_mask = pitches > 0
    
    if confidence is not None:
        # Color by confidence
        fig.add_trace(go.Scatter(
            x=times[voiced_mask],
            y=pitches[voiced_mask],
            mode='markers',
            marker=dict(
                color=confidence[voiced_mask],
                colorscale='Viridis',
                size=4,
                showscale=True,
                colorbar=dict(title='Confidence')
            ),
            name='Pitch'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=times[voiced_mask],
            y=pitches[voiced_mask],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=3),
            name='Pitch'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', type='log'),
    )
    
    return fig


def plot_chord_progression(
    chords: List[dict],
    duration: float = None,
    title: str = "Chord Progression",
    height: int = 100
) -> go.Figure:
    """
    Create a chord progression visualization.
    
    Args:
        chords: List of chord dicts with 'symbol', 'start_time', 'duration'
        duration: Total duration in beats
        title: Plot title
        height: Plot height in pixels
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if not chords:
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers'))
    else:
        if duration is None:
            duration = max(c['start_time'] + c['duration'] for c in chords)
        
        # Colors for different chord qualities
        colors = {
            'major': UI_PRIMARY_COLOR,
            'minor': UI_SECONDARY_COLOR,
            'dim': '#FFE66D',
            '7': '#95E1D3',
        }
        
        for chord in chords:
            start = chord['start_time']
            dur = chord['duration']
            symbol = chord['symbol']
            
            # Determine chord quality for color
            if 'm' in symbol.lower() and 'maj' not in symbol.lower():
                color = colors['minor']
            elif 'dim' in symbol.lower():
                color = colors['dim']
            elif '7' in symbol:
                color = colors['7']
            else:
                color = colors['major']
            
            # Add rectangle
            fig.add_shape(
                type="rect",
                x0=start,
                x1=start + dur,
                y0=0,
                y1=1,
                fillcolor=color,
                opacity=0.7,
                line=dict(color='white', width=1)
            )
            
            # Add chord label
            fig.add_annotation(
                x=start + dur / 2,
                y=0.5,
                text=symbol,
                showarrow=False,
                font=dict(size=14, color='white')
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (beats)",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            dtick=4,
            range=[0, duration] if chords else [0, 16]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[0, 1]
        ),
    )
    
    return fig

