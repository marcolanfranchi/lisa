# config.py
"""
Contains all paths, and certain constants that are used across the project.
"""

from pathlib import Path

# ---------------- PATHS ----------------
# Base directories
DATA_DIR = Path("data")
GENERATED_DIR = DATA_DIR / "generated"

# Audio Data
RAW_RECORDINGS_DIR = GENERATED_DIR / "raw_recordings"
CLEANED_RECORDINGS_DIR = GENERATED_DIR / "cleaned_recordings" 
PROCESSED_CLIPS_DIR = GENERATED_DIR / "processed_clips"
BALANCED_CLIPS_DIR = GENERATED_DIR / "balanced_clips"

# Input files
PROMPTS_FILE = DATA_DIR / "recording-prompts.json"

# Output files
MANIFEST_FILE = GENERATED_DIR / "manifest.csv"
BALANCED_MANIFEST_FILE = GENERATED_DIR / "balanced_manifest.csv"

# ---------------- SPEAKER MANAGEMENT ----------------
def get_speaker_paths(speaker_id):
    """
    Get all relevant paths for a specific speaker.
    
    Args:
        speaker_id (str): The speaker identifier
        
    Returns:
        dict: Dictionary containing all paths for the speaker
    """
    return {
        'raw_dir': RAW_RECORDINGS_DIR / speaker_id,
        'cleaned_dir': CLEANED_RECORDINGS_DIR / speaker_id,
        'clips_dir': PROCESSED_CLIPS_DIR / speaker_id,
    }

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        GENERATED_DIR,
        RAW_RECORDINGS_DIR,
        CLEANED_RECORDINGS_DIR,
        PROCESSED_CLIPS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ All directories ensured: {len(directories)} directories")

def get_all_speakers():
    """
    Get list of all speakers that have raw recordings.
    
    Returns:
        list: List of speaker IDs
    """
    if not RAW_RECORDINGS_DIR.exists():
        return []
    
    return [d.name for d in RAW_RECORDINGS_DIR.iterdir() if d.is_dir()]

# ---------------- VALIDATION ----------------
def validate_speaker_data(speaker_id):
    """
    Validate that a speaker has the expected data structure.
    
    Args:
        speaker_id (str): The speaker identifier
        
    Returns:
        dict: Validation results
    """
    paths = get_speaker_paths(speaker_id)
    
    results = {
        'speaker_id': speaker_id,
        'has_raw': paths['raw_dir'].exists(),
        'has_cleaned': paths['cleaned_dir'].exists(),
        'has_clips': paths['clips_dir'].exists(),
        'raw_files': len(list(paths['raw_dir'].glob("*.wav"))) if paths['raw_dir'].exists() else 0,
        'cleaned_files': len(list(paths['cleaned_dir'].glob("*.wav"))) if paths['cleaned_dir'].exists() else 0,
        'clip_files': len(list(paths['clips_dir'].glob("*.wav"))) if paths['clips_dir'].exists() else 0,
    }
    
    return results

# ---------------- UTILITY FUNCTIONS ----------------
def get_script_ids():
    """
    Get all script IDs from the prompts file.
    
    Returns:
        list: List of script IDs
    """
    import json
    
    if not PROMPTS_FILE.exists():
        return []
    
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    
    return [script['id'] for script in prompts]

def print_config_summary():
    """Print a summary of the current configuration."""
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 40)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Available Speakers: {len(get_all_speakers())}")
    print(f"Available Scripts: {len(get_script_ids())}")

if __name__ == "__main__":
    # Run this file directly to check configuration
    ensure_directories()
    print_config_summary()