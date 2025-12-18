# 🎵 HumToHarmony

**Music 159 - Computer Programming for Music Applications | Final Project | UC Berkeley**

---

## Purpose

**HumToHarmony** is a Python-based music production tool that transforms hummed melodies into complete musical compositions—no music theory required. It bridges the gap between having a melody "in your head" and turning it into a finished piece of music.

The application:
1. Captures your musical idea through humming
2. Detects pitch and converts it to musical notes
3. Automatically generates harmony (chords and bass lines)
4. Lets you describe sounds in plain English (e.g., "warm analog synth")
5. Synthesizes audio and exports as WAV, MP3, or MIDI

---

## Key Features & Techniques

| Feature | Description | Techniques Used |
|---------|-------------|-----------------|
| **Hum-to-Melody** | Converts humming to musical notes in ~1-2 seconds | PYIN pitch detection (fast), voice isolation bandpass filter, Krumhansl-Schmuckler key detection, note quantization |
| **Auto-Harmony** | Generates chord progressions & bass lines | Rule-based harmony generation, style templates (Pop, Jazz, Lo-Fi, Classical) |
| **Natural Language Timbre** | Describe sounds in plain English | Text parsing, descriptor-to-parameter mapping (50+ descriptors) |
| **Synthesizer Engine** | Creates professional-quality sounds | Subtractive synthesis, ADSR envelopes, oscillators (sine, saw, square, triangle) |
| **Effects & Mixing** | Polishes the final output | Reverb, delay, chorus, compression, multi-track mixing |

### Techniques from Class
- Digital signal processing (waveforms, filtering, envelopes)
- Audio analysis (pitch detection using PYIN algorithm, spectral analysis)
- MIDI processing and export
- Audio effects (comb/allpass filters for reverb, delay lines)
- Bandpass filtering for voice isolation

### Techniques Beyond Class
- Probabilistic YIN (PYIN) pitch detection optimized for monophonic voice
- Voice preprocessing (80Hz-1000Hz bandpass filter, noise gate)
- Natural language processing for timbre descriptions
- Music theory algorithms (key detection via Krumhansl-Schmuckler, chord progression generation)

---

## How to Use

### Installation

```bash
cd "final proj"
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running

```bash
streamlit run app/main.py
```
Open your browser to `http://localhost:8501`

### Workflow

1. **Upload Audio**: Record yourself humming a melody and upload the file (WAV, MP3, OGG, FLAC)
   - Supported: 5-30 second recordings
   - Best results: Clear humming in a quiet environment
2. **Analyze Melody**: Click "Analyze Melody" - takes ~1-3 seconds for typical recordings
   - Voice isolation filter removes background noise
   - PYIN algorithm detects pitch (10-20x faster than neural network methods)
   - Automatic key detection and tempo estimation
3. **Review Notes**: See your humming converted to musical notes on a piano roll
4. **Generate Harmony**: Select a style (Pop, Jazz, Lo-Fi, etc.) to auto-generate chords and bass
5. **Choose Sounds**: Type descriptions like "bright synth lead" or "warm piano", or use quick presets
6. **Export**: Download as WAV or MIDI

---

## Project Structure

```
app/
├── main.py                 # Streamlit UI
├── core/
│   ├── pitch/              # Pitch detection, key detection, quantization
│   ├── harmony/            # Chord & bass generation
│   ├── timbre/             # Natural language sound design
│   ├── synth/              # Synthesis engine (oscillators, filters, envelopes)
│   └── mixer/              # Mixing, effects, export
├── database/               # SQLite for project storage
└── utils/                  # Audio I/O, MIDI utilities
```

---

## Assignment Goal Alignment

| Goal | Implementation |
|------|----------------|
| **Creating New Sounds** | Custom synthesizer with natural language timbre control |
| **Transforming Sounds** | Pitch-to-note conversion, effects processing |
| **Classifying Sound** | Automatic key and tempo detection |
| **Supporting Creative Process** | Complete melody-to-music production pipeline |

---

## Dependencies

`librosa` (audio analysis) · `streamlit` (UI) · `numpy/scipy` (DSP) · `soundfile` (audio I/O) · `midiutil` (MIDI export)

---

*This project is for educational purposes as part of UC Berkeley's Music 159 course.*
