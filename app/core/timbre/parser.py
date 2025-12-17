"""
Natural language parser for timbre descriptions.

Extracts descriptors from user input like "warm analog synth lead".
"""

import re
from typing import List, Tuple, Set
from .descriptors import TIMBRE_DESCRIPTORS, get_descriptor_mapping


# Common filler words to ignore
STOP_WORDS = {
    "a", "an", "the", "and", "or", "with", "like", "kind", "of", 
    "sort", "type", "sounding", "sound", "tone", "timbre",
    "little", "bit", "very", "really", "quite", "somewhat",
    "that", "sounds", "is", "has", "have", "i", "want", "need",
    "give", "me", "make", "it", "please", "can", "you",
}

# Intensifier words (modify the next descriptor)
INTENSIFIERS = {
    "very": 1.3,
    "really": 1.3,
    "super": 1.5,
    "extremely": 1.5,
    "slightly": 0.7,
    "somewhat": 0.8,
    "a bit": 0.7,
    "kind of": 0.8,
    "sort of": 0.8,
}


def parse_timbre_description(description: str) -> List[Tuple[str, float]]:
    """
    Parse a natural language timbre description.
    
    Args:
        description: User's description (e.g., "warm analog synth lead")
    
    Returns:
        List of (descriptor, weight) tuples
    """
    if not description:
        return []
    
    # Normalize
    text = description.lower().strip()
    
    # Remove punctuation except hyphens
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Extract descriptors with weights
    descriptors = []
    current_weight = 1.0
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check for multi-word intensifiers
        two_word = f"{token} {tokens[i+1]}" if i + 1 < len(tokens) else ""
        
        if two_word in INTENSIFIERS:
            current_weight = INTENSIFIERS[two_word]
            i += 2
            continue
        
        if token in INTENSIFIERS:
            current_weight = INTENSIFIERS[token]
            i += 1
            continue
        
        # Skip stop words
        if token in STOP_WORDS:
            i += 1
            continue
        
        # Check if token is a known descriptor
        if token in TIMBRE_DESCRIPTORS:
            descriptors.append((token, current_weight))
            current_weight = 1.0  # Reset weight
        else:
            # Try to find partial matches or synonyms
            matched = _find_similar_descriptor(token)
            if matched:
                descriptors.append((matched, current_weight))
                current_weight = 1.0
        
        i += 1
    
    return descriptors


def extract_descriptors(description: str) -> List[str]:
    """
    Extract just the descriptor names from a description.
    
    Args:
        description: User's description
    
    Returns:
        List of descriptor names
    """
    parsed = parse_timbre_description(description)
    return [desc for desc, _ in parsed]


def _find_similar_descriptor(word: str) -> str:
    """
    Find a similar known descriptor for an unknown word.
    
    Uses simple synonym matching.
    """
    # Synonym mapping
    synonyms = {
        # Brightness synonyms
        "brilliant": "bright",
        "shiny": "bright",
        "sparkling": "bright",
        "murky": "dark",
        "muddy": "dark",
        "cloudy": "dark",
        
        # Temperature synonyms
        "hot": "warm",
        "cozy": "warm",
        "cool": "cold",
        "icy": "cold",
        "chilly": "cold",
        
        # Character synonyms
        "old": "vintage",
        "classic": "vintage",
        "new": "modern",
        "contemporary": "modern",
        "natural": "organic",
        "artificial": "synthetic",
        "electronic": "synth",
        "real": "acoustic",
        
        # Texture synonyms
        "fuzzy": "warm",
        "silky": "smooth",
        "velvet": "smooth",
        "grainy": "gritty",
        "crunchy": "gritty",
        "distorted": "dirty",
        
        # Dynamic synonyms
        "quiet": "soft",
        "loud": "aggressive",
        "powerful": "punchy",
        "weak": "soft",
        "strong": "punchy",
        "hard": "punchy",
        "percussive": "punchy",
        
        # Space synonyms
        "reverby": "wet",
        "echoey": "wet",
        "roomy": "spacious",
        "big": "wide",
        "small": "narrow",
        "huge": "thick",
        "massive": "thick",
        "tiny": "thin",
        "floaty": "ethereal",
        "heavenly": "ethereal",
        "ambient": "spacious",
        
        # Instrument synonyms
        "keyboard": "keys",
        "ep": "rhodes",
        "wurlitzer": "rhodes",
        "string": "strings",
        "violin": "strings",
        "cello": "strings",
        "orchestral": "strings",
        "synthesizer": "synth",
        "saw": "synth",
        "square": "synth",
        
        # Envelope synonyms
        "quick": "snappy",
        "fast": "snappy",
        "slow": "long",
        "sustained": "long",
        "staccato": "short",
        
        # Register synonyms
        "low": "bass",
        "bottom": "deep",
        "top": "high",
        "upper": "high",
        "lower": "deep",
    }
    
    return synonyms.get(word, None)


def get_description_summary(description: str) -> str:
    """
    Get a summary of understood descriptors from a description.
    
    Args:
        description: User's input
    
    Returns:
        Human-readable summary
    """
    parsed = parse_timbre_description(description)
    
    if not parsed:
        return "No known sound descriptors found."
    
    parts = []
    for desc, weight in parsed:
        if weight != 1.0:
            if weight > 1.0:
                parts.append(f"very {desc}")
            else:
                parts.append(f"slightly {desc}")
        else:
            parts.append(desc)
    
    return "Understood: " + ", ".join(parts)


def suggest_descriptors(partial: str) -> List[str]:
    """
    Suggest descriptors that match a partial input.
    
    Args:
        partial: Partial word typed by user
    
    Returns:
        List of matching descriptors
    """
    partial_lower = partial.lower()
    
    matches = [
        desc for desc in TIMBRE_DESCRIPTORS.keys()
        if desc.startswith(partial_lower)
    ]
    
    # Also include descriptors that contain the partial
    contains_matches = [
        desc for desc in TIMBRE_DESCRIPTORS.keys()
        if partial_lower in desc and desc not in matches
    ]
    
    return matches + contains_matches

