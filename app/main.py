"""
HumToHarmony - Main Streamlit Application

Turn your humming into complete musical compositions.
Music 159 Final Project - UC Berkeley
"""

import streamlit as st
import numpy as np
import soundfile as sf
import io
from pathlib import Path
import tempfile
import os

# Add app directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SAMPLE_RATE, DEFAULT_TEMPO, AVAILABLE_STYLES,
    ensure_directories,
)
from database import init_database, create_project, save_melody_notes, save_chords
from database.db import Note, Chord
from utils.audio_io import load_audio, normalize_audio, trim_silence
from utils.visualization import plot_waveform, plot_piano_roll, plot_chord_progression
from core.pitch import detect_pitch, extract_notes, detect_key, quantize_notes
from core.pitch.quantizer import detect_tempo
from core.harmony import generate_chords, generate_bass_line
from core.harmony.voicing import chord_to_notes
from core.harmony.styles import get_style_settings
from core.synth.engine import render_notes, SynthParams
from core.timbre import description_to_params, get_preset, get_preset_names
from core.timbre.parser import get_description_summary
from core.mixer.effects import EffectsChain
from core.mixer.tracks import Track, mix_tracks, Mixer
from core.mixer.export import export_wav, export_midi, get_audio_bytes


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="HumToHarmony",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-title {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5em;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    
    .step-header {
        color: #4ECDC4;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    
    .info-box {
        background-color: rgba(78, 205, 196, 0.1);
        border-left: 4px solid #4ECDC4;
        padding: 1em;
        border-radius: 0 8px 8px 0;
        margin: 1em 0;
    }
    
    .success-box {
        background-color: rgba(78, 205, 196, 0.2);
        border-left: 4px solid #4ECDC4;
        padding: 1em;
        border-radius: 0 8px 8px 0;
    }
    
    div[data-testid="stAudio"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Initialize
# =============================================================================

# Ensure directories exist
ensure_directories()

# Initialize database
init_database()

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = None
if 'melody_notes' not in st.session_state:
    st.session_state.melody_notes = []
if 'detected_key' not in st.session_state:
    st.session_state.detected_key = None
if 'chords' not in st.session_state:
    st.session_state.chords = []
if 'bass_notes' not in st.session_state:
    st.session_state.bass_notes = []
if 'tempo' not in st.session_state:
    st.session_state.tempo = DEFAULT_TEMPO
if 'melody_audio' not in st.session_state:
    st.session_state.melody_audio = None
if 'harmony_audio' not in st.session_state:
    st.session_state.harmony_audio = None
if 'bass_audio' not in st.session_state:
    st.session_state.bass_audio = None
if 'mixed_audio' not in st.session_state:
    st.session_state.mixed_audio = None


# =============================================================================
# Header
# =============================================================================

st.markdown('<h1 class="main-title">🎵 HumToHarmony</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Turn your humming into complete musical compositions — no music theory required</p>', unsafe_allow_html=True)


# =============================================================================
# Sidebar - Settings
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Tempo
    tempo = st.slider("Tempo (BPM)", 60, 200, int(st.session_state.tempo), 5)
    st.session_state.tempo = tempo
    
    # Style
    style = st.selectbox(
        "Musical Style",
        AVAILABLE_STYLES,
        format_func=lambda x: x.title(),
    )
    
    # Get style settings
    style_settings = get_style_settings(style)
    
    st.markdown("---")
    
    st.header("🎹 Quick Presets")
    st.caption("Quick timbre presets for each track")
    
    preset_categories = {
        "🎵 Melody": ["Classic Synth Lead", "Soft Lead", "Rhodes EP", "Bright Piano"],
        "🎹 Harmony": ["Warm Pad", "Rhodes EP", "String Ensemble", "Grand Piano"],
        "🎸 Bass": ["Sub Bass", "Synth Bass", "Acoustic Bass"],
    }
    
    for category, presets in preset_categories.items():
        st.caption(category)
    
    st.markdown("---")
    st.caption("Music 159 Final Project")
    st.caption("UC Berkeley")


# =============================================================================
# Step 1: Record Your Melody
# =============================================================================

st.markdown('<p class="step-header">Step 1: Record Your Melody</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="info-box">
    🎤 <b>Upload an audio file of your humming</b><br>
    Record yourself humming a melody using your phone or computer, then upload it here.
    Supported formats: WAV, MP3, OGG, FLAC
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your humming",
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Record yourself humming a melody and upload it here"
    )
    
    if uploaded_file is not None:
        # Save to temp file and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            audio, sr = load_audio(tmp_path, sr=SAMPLE_RATE, mono=True)
            
            # Trim silence
            audio, _, _ = trim_silence(audio)
            
            st.session_state.recording = audio
            
            st.success(f"✅ Loaded {len(audio)/SAMPLE_RATE:.1f} seconds of audio")
            
            # Show waveform
            fig = plot_waveform(audio, SAMPLE_RATE, "Your Recording")
            st.plotly_chart(fig, use_container_width=True)
            
            # Playback
            st.audio(get_audio_bytes(audio, SAMPLE_RATE), format='audio/wav')
            
        finally:
            os.unlink(tmp_path)

with col2:
    st.markdown("**Tips for best results:**")
    st.markdown("""
    - 🎵 Hum clearly and steadily
    - 🔇 Record in a quiet environment
    - ⏱️ Keep it 5-30 seconds
    - 🎯 Focus on the melody (don't worry about perfect pitch!)
    """)


# =============================================================================
# Step 2: Melody Detection
# =============================================================================

if st.session_state.recording is not None:
    st.markdown('<p class="step-header">Step 2: Your Melody</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        analyze_btn = st.button("🔍 Analyze Melody", type="primary", use_container_width=True)
    
    if analyze_btn or len(st.session_state.melody_notes) > 0:
        
        if analyze_btn:
            with st.spinner("Detecting pitch... (this may take a moment)"):
                # Detect pitch
                pitch_result = detect_pitch(st.session_state.recording, SAMPLE_RATE)
                
                # Extract notes
                notes = extract_notes(pitch_result)
                
                # Detect tempo
                detected_tempo, _ = detect_tempo(st.session_state.recording, SAMPLE_RATE, tempo)
                st.session_state.tempo = detected_tempo
                
                # Convert to beat time
                from core.pitch.detector import notes_to_beats
                notes_in_beats = notes_to_beats(notes, st.session_state.tempo)
                
                # Detect key
                key_result = detect_key(notes_in_beats)
                st.session_state.detected_key = key_result
                
                # Quantize notes
                quantized = quantize_notes(
                    notes_in_beats,
                    tempo=st.session_state.tempo,
                    key=key_result.key,
                    scale=key_result.scale,
                )
                
                st.session_state.melody_notes = quantized
        
        # Display results
        if st.session_state.melody_notes:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Detected Key", f"{st.session_state.detected_key.key} {st.session_state.detected_key.scale}")
            with col2:
                st.metric("Tempo", f"{st.session_state.tempo:.0f} BPM")
            with col3:
                st.metric("Notes Found", len(st.session_state.melody_notes))
            
            # Piano roll visualization
            notes_for_plot = [n.to_dict() for n in st.session_state.melody_notes]
            fig = plot_piano_roll(notes_for_plot, title="Detected Melody")
            st.plotly_chart(fig, use_container_width=True)
            
            # Manual key override
            col1, col2 = st.columns(2)
            with col1:
                override_key = st.selectbox(
                    "Override Key (optional)",
                    ["Auto"] + ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                )
            with col2:
                override_scale = st.selectbox(
                    "Scale",
                    ["major", "minor"],
                )
            
            if override_key != "Auto":
                st.session_state.detected_key.key = override_key
                st.session_state.detected_key.scale = override_scale


# =============================================================================
# Step 3: Harmony Generation
# =============================================================================

if len(st.session_state.melody_notes) > 0:
    st.markdown('<p class="step-header">Step 3: Generate Harmony</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        generate_btn = st.button("🎹 Generate Chords", type="primary", use_container_width=True)
    
    if generate_btn or len(st.session_state.chords) > 0:
        
        if generate_btn:
            with st.spinner("Generating harmony..."):
                # Generate chord progression
                chord_prog = generate_chords(
                    st.session_state.melody_notes,
                    key=st.session_state.detected_key.key,
                    scale=st.session_state.detected_key.scale,
                    style=style,
                )
                st.session_state.chords = chord_prog.chords
                
                # Generate bass line
                bass_notes = generate_bass_line(
                    chord_prog.chords,
                    style=style,
                )
                st.session_state.bass_notes = bass_notes
        
        if st.session_state.chords:
            # Display chord progression
            chord_symbols = " → ".join([c.symbol for c in st.session_state.chords[:8]])
            st.info(f"**Chord Progression:** {chord_symbols}")
            
            # Chord visualization
            chords_for_plot = [c.to_dict() for c in st.session_state.chords]
            fig = plot_chord_progression(chords_for_plot, title="Generated Chords")
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Step 4: Choose Your Sounds
# =============================================================================

if len(st.session_state.chords) > 0:
    st.markdown('<p class="step-header">Step 4: Choose Your Sounds</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    🗣️ <b>Describe the sound you want in plain English!</b><br>
    Try descriptions like: "warm analog synth", "bright piano", "dreamy pad", "punchy bass"
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎵 Melody Sound**")
        melody_desc = st.text_input(
            "Describe melody sound",
            value=style_settings.melody_timbre,
            label_visibility="collapsed",
            placeholder="e.g., bright synth lead",
        )
        melody_preset = st.selectbox(
            "Or choose preset",
            ["Custom"] + ["Classic Synth Lead", "Soft Lead", "Rhodes EP", "Grand Piano", "Pluck"],
            key="melody_preset",
        )
        if melody_preset != "Custom":
            melody_desc = melody_preset.lower()
        st.caption(get_description_summary(melody_desc))
    
    with col2:
        st.markdown("**🎹 Harmony Sound**")
        harmony_desc = st.text_input(
            "Describe harmony sound",
            value=style_settings.harmony_timbre,
            label_visibility="collapsed",
            placeholder="e.g., warm Rhodes piano",
        )
        harmony_preset = st.selectbox(
            "Or choose preset",
            ["Custom"] + ["Warm Pad", "Ethereal Pad", "Rhodes EP", "String Ensemble", "Grand Piano"],
            key="harmony_preset",
        )
        if harmony_preset != "Custom":
            harmony_desc = harmony_preset.lower()
        st.caption(get_description_summary(harmony_desc))
    
    with col3:
        st.markdown("**🎸 Bass Sound**")
        bass_desc = st.text_input(
            "Describe bass sound",
            value=style_settings.bass_timbre,
            label_visibility="collapsed",
            placeholder="e.g., deep sub bass",
        )
        bass_preset = st.selectbox(
            "Or choose preset",
            ["Custom"] + ["Sub Bass", "Synth Bass", "Wobble Bass", "Acoustic Bass"],
            key="bass_preset",
        )
        if bass_preset != "Custom":
            bass_desc = bass_preset.lower()
        st.caption(get_description_summary(bass_desc))
    
    # Render button
    st.markdown("")
    render_btn = st.button("🎧 Render Audio", type="primary", use_container_width=True)
    
    if render_btn:
        with st.spinner("Synthesizing your music... 🎵"):
            # Get synth params from descriptions
            melody_params = description_to_params(melody_desc)
            harmony_params = description_to_params(harmony_desc)
            bass_params = description_to_params(bass_desc)
            
            # Render melody
            melody_audio = render_notes(
                st.session_state.melody_notes,
                melody_params,
                st.session_state.tempo,
                SAMPLE_RATE,
            )
            st.session_state.melody_audio = melody_audio
            
            # Render harmony (convert chords to notes)
            harmony_notes = []
            for chord in st.session_state.chords:
                chord_notes = chord_to_notes(chord, velocity=65, style=style)
                harmony_notes.extend(chord_notes)
            
            harmony_audio = render_notes(
                harmony_notes,
                harmony_params,
                st.session_state.tempo,
                SAMPLE_RATE,
            )
            st.session_state.harmony_audio = harmony_audio
            
            # Render bass
            bass_audio = render_notes(
                st.session_state.bass_notes,
                bass_params,
                st.session_state.tempo,
                SAMPLE_RATE,
            )
            st.session_state.bass_audio = bass_audio
            
            # Mix tracks
            melody_track = Track("Melody", melody_audio, volume=0.8)
            harmony_track = Track("Harmony", harmony_audio, volume=0.5, pan=-0.2)
            bass_track = Track("Bass", bass_audio, volume=0.7, pan=0.0)
            
            # Add effects based on style
            melody_track.effects = EffectsChain(reverb=0.2, delay=0.1)
            harmony_track.effects = EffectsChain(reverb=0.35, chorus=0.2)
            bass_track.effects = EffectsChain(reverb=0.1)
            
            mixed = mix_tracks([melody_track, harmony_track, bass_track], SAMPLE_RATE)
            
            # Convert to mono for preview
            mixed_mono = (mixed[0] + mixed[1]) / 2
            st.session_state.mixed_audio = mixed_mono
        
        st.success("✅ Audio rendered successfully!")


# =============================================================================
# Step 5: Preview & Export
# =============================================================================

if st.session_state.mixed_audio is not None:
    st.markdown('<p class="step-header">Step 5: Preview & Export</p>', unsafe_allow_html=True)
    
    # Mixer
    st.markdown("**🎚️ Mix**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        melody_vol = st.slider("Melody", 0.0, 1.0, 0.8, 0.05)
    with col2:
        harmony_vol = st.slider("Harmony", 0.0, 1.0, 0.5, 0.05)
    with col3:
        bass_vol = st.slider("Bass", 0.0, 1.0, 0.7, 0.05)
    
    # Remix if volumes changed
    if st.button("🔄 Update Mix"):
        melody_track = Track("Melody", st.session_state.melody_audio, volume=melody_vol)
        harmony_track = Track("Harmony", st.session_state.harmony_audio, volume=harmony_vol, pan=-0.2)
        bass_track = Track("Bass", st.session_state.bass_audio, volume=bass_vol)
        
        melody_track.effects = EffectsChain(reverb=0.2, delay=0.1)
        harmony_track.effects = EffectsChain(reverb=0.35, chorus=0.2)
        bass_track.effects = EffectsChain(reverb=0.1)
        
        mixed = mix_tracks([melody_track, harmony_track, bass_track], SAMPLE_RATE)
        mixed_mono = (mixed[0] + mixed[1]) / 2
        st.session_state.mixed_audio = mixed_mono
    
    # Preview
    st.markdown("**🎧 Preview**")
    st.audio(get_audio_bytes(st.session_state.mixed_audio, SAMPLE_RATE), format='audio/wav')
    
    # Waveform
    fig = plot_waveform(st.session_state.mixed_audio, SAMPLE_RATE, "Final Mix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.markdown("**💾 Export**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wav_bytes = get_audio_bytes(st.session_state.mixed_audio, SAMPLE_RATE, 'wav')
        st.download_button(
            "📥 Download WAV",
            wav_bytes,
            "hum_to_harmony_mix.wav",
            "audio/wav",
            use_container_width=True,
        )
    
    with col2:
        # MIDI export
        if st.session_state.melody_notes:
            midi_buffer = io.BytesIO()
            from midiutil import MIDIFile
            midi = MIDIFile(1)
            midi.addTempo(0, 0, st.session_state.tempo)
            for note in st.session_state.melody_notes:
                midi.addNote(0, 0, note.pitch, note.start_time, note.duration, note.velocity)
            midi.writeFile(midi_buffer)
            midi_buffer.seek(0)
            
            st.download_button(
                "📥 Download MIDI",
                midi_buffer.getvalue(),
                "hum_to_harmony_melody.mid",
                "audio/midi",
                use_container_width=True,
            )
    
    with col3:
        st.info("🎉 Your music is ready!")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2em;">
    <p>🎵 <b>HumToHarmony</b> - Music 159 Final Project</p>
    <p>Turn your humming into complete musical compositions</p>
    <p style="font-size: 0.8em;">UC Berkeley | Fall 2024</p>
</div>
""", unsafe_allow_html=True)

