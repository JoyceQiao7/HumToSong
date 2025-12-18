"""
Configuration settings for HumToHarmony.
"""

from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
PROJECTS_DIR = DATA_DIR / "projects"
PRESETS_DIR = DATA_DIR / "presets"
CACHE_DIR = DATA_DIR / "cache"

# Database file
DATABASE_PATH = DB_DIR / "hum_to_harmony.db"

# =============================================================================
# Audio Configuration
# =============================================================================

SAMPLE_RATE = 44100          # Standard audio sample rate
BIT_DEPTH = 16               # Audio bit depth
CHANNELS = 1                 # Mono for recording (stereo for output)
BUFFER_SIZE = 1024           # Audio buffer size

# Recording settings
MAX_RECORDING_DURATION = 60  # Maximum recording length in seconds
SILENCE_THRESHOLD = 0.01     # Threshold for silence detection

# =============================================================================
# Music Configuration
# =============================================================================

DEFAULT_TEMPO = 120          # BPM
DEFAULT_TIME_SIGNATURE = (4, 4)
DEFAULT_KEY = "C"
DEFAULT_SCALE = "major"

# Note quantization
QUANTIZE_RESOLUTION = 16     # 1/16th notes
MIN_NOTE_DURATION = 0.125    # Minimum note duration in beats

# Pitch detection
PITCH_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for pitch detection
PITCH_METHOD = "pyin"        # Primary method: 'pyin' (fast) or 'crepe' (slow but accurate)
PITCH_MODEL = "tiny"         # CREPE model size if using CREPE: 'tiny', 'small', 'medium', 'large', 'full'

# =============================================================================
# Harmony Generation
# =============================================================================

AVAILABLE_STYLES = ["pop", "jazz", "lofi", "classical", "electronic"]
DEFAULT_STYLE = "pop"

# Chord progression templates by style
CHORD_TEMPLATES = {
    "pop": [
        ["I", "V", "vi", "IV"],
        ["I", "IV", "V", "I"],
        ["vi", "IV", "I", "V"],
        ["I", "vi", "IV", "V"],
    ],
    "jazz": [
        ["ii", "V", "I", "I"],
        ["I", "vi", "ii", "V"],
        ["iii", "vi", "ii", "V"],
        ["I", "IV", "iii", "vi"],
    ],
    "lofi": [
        ["ii", "V", "I", "vi"],
        ["I", "V", "vi", "iii"],
        ["vi", "ii", "V", "I"],
    ],
    "classical": [
        ["I", "IV", "V", "I"],
        ["I", "ii", "V", "I"],
        ["I", "vi", "IV", "V"],
    ],
    "electronic": [
        ["i", "VI", "III", "VII"],
        ["i", "iv", "VI", "V"],
        ["vi", "IV", "I", "V"],
    ],
}

# =============================================================================
# Synthesis Configuration
# =============================================================================

# ADSR envelope defaults (in seconds)
DEFAULT_ATTACK = 0.01
DEFAULT_DECAY = 0.1
DEFAULT_SUSTAIN = 0.7        # Sustain level (0-1)
DEFAULT_RELEASE = 0.3

# Oscillator settings
AVAILABLE_WAVEFORMS = ["sine", "triangle", "sawtooth", "square", "noise"]
DEFAULT_WAVEFORM = "sawtooth"

# Filter settings
DEFAULT_FILTER_CUTOFF = 0.5  # Normalized (0-1)
DEFAULT_FILTER_RESONANCE = 0.2

# =============================================================================
# UI Configuration
# =============================================================================

# Color scheme
UI_PRIMARY_COLOR = "#FF6B6B"
UI_SECONDARY_COLOR = "#4ECDC4"
UI_BACKGROUND_COLOR = "#1A1A2E"
UI_TEXT_COLOR = "#EAEAEA"

# Piano roll display
PIANO_ROLL_HEIGHT = 400
PIANO_ROLL_NOTE_RANGE = (36, 84)  # C2 to C6

# =============================================================================
# Create directories if they don't exist
# =============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    for directory in [DATA_DIR, DB_DIR, PROJECTS_DIR, PRESETS_DIR, CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create preset subdirectories
    (PRESETS_DIR / "instruments").mkdir(exist_ok=True)
    (PRESETS_DIR / "styles").mkdir(exist_ok=True)

