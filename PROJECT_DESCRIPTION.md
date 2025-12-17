# HumToHarmony: Project Description

**Music 159 - Computer Programming for Music Applications**  
**Final Project**  
**UC Berkeley**

---

## Purpose

**HumToHarmony** is a Python-based music production tool designed to democratize music creation for people who don't know music theory. The application transforms simple hummed melodies into complete musical compositions by:

1. **Capturing the user's musical idea** through humming or singing
2. **Detecting the pitch** and converting it to musical notes
3. **Automatically generating harmony** (chord progressions and bass lines) that fit the melody
4. **Allowing natural language timbre selection** — users describe the sound they want in plain English (e.g., "warm analog synth", "bright piano")
5. **Synthesizing professional-quality audio** and exporting as WAV, MP3, or MIDI

The project solves a real pain point for musicians and non-musicians alike: the gap between having a melody "in your head" and turning it into a complete piece of music.

---

## Key Features

### 1. Hum-to-Melody Conversion
- Uses CREPE neural network for state-of-the-art monophonic pitch detection
- Converts continuous pitch contour to discrete musical notes
- Automatic key detection using Krumhansl-Schmuckler algorithm
- Note quantization to musical grid (1/4, 1/8, 1/16 notes)

### 2. Automatic Harmony Generation
- Rule-based chord progression generation
- Style-aware harmony (Pop, Jazz, Lo-Fi, Classical, Electronic)
- Automatic bass line generation with style-appropriate patterns
- Intelligent chord voicing with voice leading

### 3. Natural Language Timbre System
- **Novel feature**: Describe sounds in plain English
- Maps descriptive words (warm, bright, punchy, dreamy, etc.) to synthesizer parameters
- 50+ descriptor vocabulary covering tone, texture, dynamics, and space
- Quick presets for common instrument sounds

### 4. Custom Synthesizer Engine
- Subtractive synthesis with multiple oscillator types (sine, saw, square, triangle)
- ADSR amplitude and filter envelopes
- Unison/detune for thick sounds
- Built-in effects: reverb, delay, chorus, compression

### 5. Professional Mixing & Export
- Multi-track mixing with volume and pan controls
- Master effects processing
- Export to WAV, MP3, and MIDI formats

---

## Techniques Used

### From Class
- **Digital Signal Processing**: Waveform generation, filtering, envelopes
- **Audio Analysis**: Pitch detection, onset detection, spectral analysis
- **MIDI Processing**: Note representation, file export
- **Audio Effects**: Reverb (comb/allpass filters), delay, compression

### Beyond Class
- **Machine Learning**: CREPE neural network for pitch detection
- **Natural Language Processing**: Text parsing for timbre descriptions
- **Music Theory Algorithms**: Key detection, chord progression generation
- **Vector-based Sound Design**: Mapping descriptors to synthesis parameters

---

## How to Use

### Installation

```bash
# Navigate to project directory
cd "final proj"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app/main.py
```

Then open your browser to `http://localhost:8501`

### Workflow

1. **Record your melody**: Use your phone or computer to record yourself humming, then upload the audio file.

2. **Review detected notes**: See your humming converted to musical notes on a piano roll. The app automatically detects the key.

3. **Generate harmony**: Choose a musical style and let the app generate chords and bass lines.

4. **Describe your sounds**: Type natural language descriptions for each instrument:
   - Melody: "bright, playful synth lead"
   - Harmony: "warm Rhodes electric piano"
   - Bass: "deep, smooth sub bass"

5. **Export**: Download your creation as WAV, MP3, or MIDI.

---

## Project Structure

```
hum_to_harmony/
├── app/
│   ├── main.py              # Streamlit UI entry point
│   ├── config.py            # Configuration settings
│   ├── database/            # SQLite database operations
│   ├── core/
│   │   ├── pitch/           # Pitch detection & key detection
│   │   ├── harmony/         # Chord & bass generation
│   │   ├── timbre/          # Natural language sound design
│   │   ├── synth/           # Sound synthesis engine
│   │   └── mixer/           # Audio mixing & export
│   └── utils/               # Audio I/O, MIDI, visualization
├── data/
│   ├── db/                  # SQLite database
│   ├── projects/            # User project files
│   └── presets/             # Sound preset definitions
├── requirements.txt
├── README.md
└── PROJECT_DESCRIPTION.md
```

---

## Technical Highlights

### Pitch Detection Pipeline
```
Raw Audio → CREPE Neural Network → Pitch Contour → Voiced Segmentation
    → Note Extraction → Key Detection → Quantization → Musical Notes
```

### Natural Language Timbre Mapping
```
"warm analog synth" → Parser → Descriptors: [warm, analog, synth]
    → Parameter Mapping → SynthParams(waveform='sawtooth', 
       filter_cutoff=0.5, detune=5, noise=0.02, ...)
```

### Harmony Generation
```
Melody Notes → Key Analysis → Scale Degree Detection
    → Style Templates → Chord Progression → Bass Pattern
```

---

## Dependencies

- **librosa**: Audio analysis and feature extraction
- **crepe**: Neural network pitch detection
- **streamlit**: Web-based user interface
- **numpy/scipy**: Signal processing
- **soundfile**: Audio I/O
- **midiutil**: MIDI file generation

---

## Alignment with Assignment Goals

| Goal | Implementation |
|------|----------------|
| **Creating New Sounds** | Custom synthesizer with NL timbre control |
| **Transforming Sounds** | Pitch-to-note conversion, effects processing |
| **Classifying Sound** | Key detection, tempo detection |
| **Supporting Creative Process** | Complete melody-to-music pipeline |

---

## Future Improvements

- Real-time audio recording directly in browser
- More sophisticated harmony algorithms (jazz voicings, modulations)
- Sample-based synthesis for more realistic instrument sounds
- AI-powered arrangement suggestions
- Collaborative features for sharing projects

---

## Acknowledgments

- CREPE: A Convolutional Representation for Pitch Estimation (Kim et al., 2018)
- librosa: Audio and Music Signal Analysis in Python
- Streamlit: The fastest way to build data apps

