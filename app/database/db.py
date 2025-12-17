"""
SQLite database operations for HumToHarmony.

This module provides all database CRUD operations for managing:
- Projects (user sessions)
- Tracks (melody, harmony, bass)
- Notes (MIDI note data)
- Chords (chord progressions)
- Timbres (sound settings)
- Recordings (raw audio paths)
"""

import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH, DB_DIR


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Note:
    """Represents a single musical note."""
    pitch: int           # MIDI note number (0-127)
    start_time: float    # Start time in beats
    duration: float      # Duration in beats
    velocity: int = 100  # MIDI velocity (0-127)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chord:
    """Represents a chord in the progression."""
    symbol: str          # e.g., "Cmaj7", "Am", "G"
    start_time: float    # Start time in beats
    duration: float      # Duration in beats
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Timbre:
    """Represents timbre/sound settings for a track."""
    description: Optional[str] = None  # Natural language description
    preset_name: Optional[str] = None  # Preset name if using preset
    parameters: Optional[Dict] = None  # Synth parameters
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Database Connection
# =============================================================================

@contextmanager
def get_connection():
    """
    Context manager for database connections.
    Ensures proper commit/rollback and connection cleanup.
    """
    # Ensure directory exists
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# =============================================================================
# Database Initialization
# =============================================================================

def init_database():
    """
    Initialize the database with all required tables.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    with get_connection() as conn:
        conn.executescript("""
            -- Projects table: Each humming session becomes a project
            CREATE TABLE IF NOT EXISTS projects (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tempo           REAL DEFAULT 120.0,
                key_signature   TEXT,
                time_signature  TEXT DEFAULT '4/4',
                style           TEXT DEFAULT 'pop'
            );
            
            -- Tracks table: melody, harmony, bass tracks per project
            CREATE TABLE IF NOT EXISTS tracks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id      INTEGER NOT NULL,
                track_type      TEXT NOT NULL,
                audio_path      TEXT,
                volume          REAL DEFAULT 0.8,
                pan             REAL DEFAULT 0.0,
                muted           INTEGER DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            
            -- Notes table: MIDI notes for each track
            CREATE TABLE IF NOT EXISTS notes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id        INTEGER NOT NULL,
                pitch           INTEGER NOT NULL,
                start_time      REAL NOT NULL,
                duration        REAL NOT NULL,
                velocity        INTEGER DEFAULT 100,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );
            
            -- Chords table: Chord progressions per project
            CREATE TABLE IF NOT EXISTS chords (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id      INTEGER NOT NULL,
                chord_symbol    TEXT NOT NULL,
                start_time      REAL NOT NULL,
                duration        REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            
            -- Timbres table: Sound settings for each track
            CREATE TABLE IF NOT EXISTS timbres (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id        INTEGER NOT NULL UNIQUE,
                description     TEXT,
                preset_name     TEXT,
                parameters      TEXT,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );
            
            -- Recordings table: Raw audio file references
            CREATE TABLE IF NOT EXISTS recordings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id      INTEGER NOT NULL,
                audio_path      TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration        REAL,
                sample_rate     INTEGER DEFAULT 44100,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            
            -- Create indexes for faster queries
            CREATE INDEX IF NOT EXISTS idx_tracks_project ON tracks(project_id);
            CREATE INDEX IF NOT EXISTS idx_notes_track ON notes(track_id);
            CREATE INDEX IF NOT EXISTS idx_chords_project ON chords(project_id);
        """)
    
    print(f"Database initialized at: {DATABASE_PATH}")


# =============================================================================
# Project Operations
# =============================================================================

def create_project(name: str, tempo: float = 120.0, style: str = "pop") -> int:
    """
    Create a new project and return its ID.
    
    Args:
        name: Project name
        tempo: Tempo in BPM
        style: Musical style (pop, jazz, lofi, etc.)
    
    Returns:
        The new project's ID
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO projects (name, tempo, style) 
               VALUES (?, ?, ?)""",
            (name, tempo, style)
        )
        project_id = cursor.lastrowid
        
        # Create default tracks for the project
        for track_type in ["melody", "harmony", "bass"]:
            conn.execute(
                """INSERT INTO tracks (project_id, track_type) 
                   VALUES (?, ?)""",
                (project_id, track_type)
            )
        
        return project_id


def get_project(project_id: int) -> Optional[Dict[str, Any]]:
    """
    Load a project with all its tracks.
    
    Args:
        project_id: The project ID to load
    
    Returns:
        Dictionary with project data and tracks, or None if not found
    """
    with get_connection() as conn:
        project = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        
        if not project:
            return None
        
        tracks = conn.execute(
            "SELECT * FROM tracks WHERE project_id = ?", (project_id,)
        ).fetchall()
        
        return {
            "project": dict(project),
            "tracks": {t["track_type"]: dict(t) for t in tracks}
        }


def get_all_projects() -> List[Dict[str, Any]]:
    """
    Get all projects (summary info only).
    
    Returns:
        List of project dictionaries
    """
    with get_connection() as conn:
        projects = conn.execute(
            """SELECT id, name, created_at, tempo, key_signature, style 
               FROM projects ORDER BY updated_at DESC"""
        ).fetchall()
        return [dict(p) for p in projects]


def update_project(project_id: int, **kwargs) -> bool:
    """
    Update project fields.
    
    Args:
        project_id: Project to update
        **kwargs: Fields to update (name, tempo, key_signature, style, etc.)
    
    Returns:
        True if updated, False if project not found
    """
    if not kwargs:
        return False
    
    # Build dynamic UPDATE query
    fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [project_id]
    
    with get_connection() as conn:
        cursor = conn.execute(
            f"""UPDATE projects 
                SET {fields}, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?""",
            values
        )
        return cursor.rowcount > 0


def delete_project(project_id: int) -> bool:
    """
    Delete a project and all associated data.
    
    Args:
        project_id: Project to delete
    
    Returns:
        True if deleted, False if not found
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM projects WHERE id = ?", (project_id,)
        )
        return cursor.rowcount > 0


# =============================================================================
# Note Operations
# =============================================================================

def save_melody_notes(project_id: int, notes: List[Note]) -> int:
    """
    Save detected melody notes to the project's melody track.
    
    Args:
        project_id: Project ID
        notes: List of Note objects
    
    Returns:
        Track ID where notes were saved
    """
    with get_connection() as conn:
        # Get or create melody track
        track = conn.execute(
            """SELECT id FROM tracks 
               WHERE project_id = ? AND track_type = 'melody'""",
            (project_id,)
        ).fetchone()
        
        if track:
            track_id = track["id"]
            # Clear existing notes
            conn.execute("DELETE FROM notes WHERE track_id = ?", (track_id,))
        else:
            cursor = conn.execute(
                """INSERT INTO tracks (project_id, track_type) 
                   VALUES (?, 'melody')""",
                (project_id,)
            )
            track_id = cursor.lastrowid
        
        # Insert new notes
        conn.executemany(
            """INSERT INTO notes (track_id, pitch, start_time, duration, velocity)
               VALUES (?, ?, ?, ?, ?)""",
            [(track_id, n.pitch, n.start_time, n.duration, n.velocity) for n in notes]
        )
        
        return track_id


def get_melody_notes(project_id: int) -> List[Note]:
    """
    Get all melody notes for a project.
    
    Args:
        project_id: Project ID
    
    Returns:
        List of Note objects
    """
    with get_connection() as conn:
        notes = conn.execute(
            """SELECT n.pitch, n.start_time, n.duration, n.velocity
               FROM notes n
               JOIN tracks t ON n.track_id = t.id
               WHERE t.project_id = ? AND t.track_type = 'melody'
               ORDER BY n.start_time""",
            (project_id,)
        ).fetchall()
        
        return [Note(**dict(n)) for n in notes]


def save_track_notes(project_id: int, track_type: str, notes: List[Note]) -> int:
    """
    Save notes to a specific track type (harmony, bass, etc.).
    
    Args:
        project_id: Project ID
        track_type: Type of track ('melody', 'harmony', 'bass')
        notes: List of Note objects
    
    Returns:
        Track ID where notes were saved
    """
    with get_connection() as conn:
        # Get or create track
        track = conn.execute(
            """SELECT id FROM tracks 
               WHERE project_id = ? AND track_type = ?""",
            (project_id, track_type)
        ).fetchone()
        
        if track:
            track_id = track["id"]
            conn.execute("DELETE FROM notes WHERE track_id = ?", (track_id,))
        else:
            cursor = conn.execute(
                """INSERT INTO tracks (project_id, track_type) 
                   VALUES (?, ?)""",
                (project_id, track_type)
            )
            track_id = cursor.lastrowid
        
        conn.executemany(
            """INSERT INTO notes (track_id, pitch, start_time, duration, velocity)
               VALUES (?, ?, ?, ?, ?)""",
            [(track_id, n.pitch, n.start_time, n.duration, n.velocity) for n in notes]
        )
        
        return track_id


def get_track_notes(project_id: int, track_type: str) -> List[Note]:
    """
    Get notes for a specific track type.
    
    Args:
        project_id: Project ID
        track_type: Type of track
    
    Returns:
        List of Note objects
    """
    with get_connection() as conn:
        notes = conn.execute(
            """SELECT n.pitch, n.start_time, n.duration, n.velocity
               FROM notes n
               JOIN tracks t ON n.track_id = t.id
               WHERE t.project_id = ? AND t.track_type = ?
               ORDER BY n.start_time""",
            (project_id, track_type)
        ).fetchall()
        
        return [Note(**dict(n)) for n in notes]


# =============================================================================
# Chord Operations
# =============================================================================

def save_chords(project_id: int, chords: List[Chord]):
    """
    Save chord progression to project.
    
    Args:
        project_id: Project ID
        chords: List of Chord objects
    """
    with get_connection() as conn:
        # Clear existing chords
        conn.execute("DELETE FROM chords WHERE project_id = ?", (project_id,))
        
        # Insert new chords
        conn.executemany(
            """INSERT INTO chords (project_id, chord_symbol, start_time, duration)
               VALUES (?, ?, ?, ?)""",
            [(project_id, c.symbol, c.start_time, c.duration) for c in chords]
        )


def get_chords(project_id: int) -> List[Chord]:
    """
    Get chord progression for a project.
    
    Args:
        project_id: Project ID
    
    Returns:
        List of Chord objects
    """
    with get_connection() as conn:
        chords = conn.execute(
            """SELECT chord_symbol as symbol, start_time, duration
               FROM chords WHERE project_id = ?
               ORDER BY start_time""",
            (project_id,)
        ).fetchall()
        
        return [Chord(**dict(c)) for c in chords]


# =============================================================================
# Timbre Operations
# =============================================================================

def save_timbre(project_id: int, track_type: str, timbre: Timbre):
    """
    Save timbre settings for a track.
    
    Args:
        project_id: Project ID
        track_type: Track type ('melody', 'harmony', 'bass')
        timbre: Timbre object with settings
    """
    with get_connection() as conn:
        # Get track ID
        track = conn.execute(
            """SELECT id FROM tracks 
               WHERE project_id = ? AND track_type = ?""",
            (project_id, track_type)
        ).fetchone()
        
        if not track:
            raise ValueError(f"Track '{track_type}' not found for project {project_id}")
        
        track_id = track["id"]
        params_json = json.dumps(timbre.parameters) if timbre.parameters else None
        
        # Insert or replace timbre
        conn.execute(
            """INSERT OR REPLACE INTO timbres 
               (track_id, description, preset_name, parameters)
               VALUES (?, ?, ?, ?)""",
            (track_id, timbre.description, timbre.preset_name, params_json)
        )


def get_timbre(project_id: int, track_type: str) -> Optional[Timbre]:
    """
    Get timbre settings for a track.
    
    Args:
        project_id: Project ID
        track_type: Track type
    
    Returns:
        Timbre object or None if not set
    """
    with get_connection() as conn:
        result = conn.execute(
            """SELECT t.description, t.preset_name, t.parameters
               FROM timbres t
               JOIN tracks tr ON t.track_id = tr.id
               WHERE tr.project_id = ? AND tr.track_type = ?""",
            (project_id, track_type)
        ).fetchone()
        
        if not result:
            return None
        
        params = json.loads(result["parameters"]) if result["parameters"] else None
        return Timbre(
            description=result["description"],
            preset_name=result["preset_name"],
            parameters=params
        )


# =============================================================================
# Recording Operations
# =============================================================================

def save_recording_path(project_id: int, audio_path: str, 
                        duration: float = None, sample_rate: int = 44100) -> int:
    """
    Save a recording reference to the project.
    
    Args:
        project_id: Project ID
        audio_path: Path to the audio file
        duration: Audio duration in seconds
        sample_rate: Sample rate of the recording
    
    Returns:
        Recording ID
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO recordings (project_id, audio_path, duration, sample_rate)
               VALUES (?, ?, ?, ?)""",
            (project_id, audio_path, duration, sample_rate)
        )
        return cursor.lastrowid


def get_recording_path(project_id: int) -> Optional[str]:
    """
    Get the most recent recording path for a project.
    
    Args:
        project_id: Project ID
    
    Returns:
        Audio file path or None
    """
    with get_connection() as conn:
        result = conn.execute(
            """SELECT audio_path FROM recordings 
               WHERE project_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (project_id,)
        ).fetchone()
        
        return result["audio_path"] if result else None


# =============================================================================
# Initialization
# =============================================================================

# Initialize database when module is imported
if __name__ == "__main__":
    init_database()
    print("Database initialized successfully!")

