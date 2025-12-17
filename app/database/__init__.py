"""
Database module for HumToHarmony.
Handles SQLite operations for projects, tracks, and notes.
"""

from .db import (
    init_database,
    get_connection,
    create_project,
    get_project,
    get_all_projects,
    delete_project,
    save_melody_notes,
    get_melody_notes,
    save_chords,
    get_chords,
    save_timbre,
    get_timbre,
    save_recording_path,
    update_project,
)

__all__ = [
    "init_database",
    "get_connection",
    "create_project",
    "get_project",
    "get_all_projects",
    "delete_project",
    "save_melody_notes",
    "get_melody_notes",
    "save_chords",
    "get_chords",
    "save_timbre",
    "get_timbre",
    "save_recording_path",
    "update_project",
]

