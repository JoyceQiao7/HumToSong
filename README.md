# 🎵 HumToHarmony

**Turn your humming into complete musical compositions — no music theory required.**

HumToHarmony is a Python-based music production tool that transforms simple hummed melodies into full arrangements with harmony, bass, and professional-quality sounds. Describe the sound you want in plain English, and the app brings your musical ideas to life.

## 🎯 Features

- **🎤 Hum-to-Melody**: Record your humming and watch it transform into precise musical notes
- **🎹 Auto-Harmony**: Automatically generates chord progressions that fit your melody
- **🎸 Smart Bass Lines**: Creates bass lines that groove with your music
- **🗣️ Natural Language Timbre**: Describe sounds in plain English ("warm analog synth", "bright piano")
- **🎨 Multiple Styles**: Pop, Jazz, Lo-Fi, Classical, Electronic
- **💾 Export**: Save as WAV, MP3, or MIDI

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd "final proj"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Running the App

```bash
streamlit run app/main.py
```

Then open your browser to `http://localhost:8501`

## 📖 How to Use

### Step 1: Record Your Melody
Click the record button and hum your melody idea. Don't worry about being perfectly in tune — the app will help!

### Step 2: Review Your Notes
See your humming converted to musical notes on a piano roll. The app detects the key automatically.

### Step 3: Generate Harmony
Choose a musical style (Pop, Jazz, Lo-Fi, etc.) and let the app generate chords and bass.

### Step 4: Choose Your Sounds
Describe the sound you want in plain English:
- Melody: *"bright, playful synth lead"*
- Harmony: *"warm Rhodes electric piano"*
- Bass: *"deep, smooth sub bass"*

Or choose from quick presets!

### Step 5: Export
Preview your creation and export as WAV, MP3, or MIDI.

## 🏗️ Project Structure

```
hum_to_harmony/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── config.py            # Configuration
│   ├── ui/                  # UI components
│   ├── core/                # Core logic
│   │   ├── pitch/           # Pitch detection
│   │   ├── harmony/         # Chord & bass generation
│   │   ├── timbre/          # NL timbre system
│   │   ├── synth/           # Sound synthesis
│   │   └── mixer/           # Audio mixing
│   ├── database/            # SQLite operations
│   └── utils/               # Utilities
├── data/
│   ├── db/                  # SQLite database
│   ├── projects/            # User projects
│   └── presets/             # Sound presets
├── requirements.txt
└── README.md
```

## 🎓 Course Information

**Course**: Music 159 - Computer Programming for Music Applications  
**Assignment**: Final Project  
**University**: UC Berkeley

## 📝 License

This project is for educational purposes as part of UC Berkeley's Music 159 course.

