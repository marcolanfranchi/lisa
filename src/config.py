"""Configuration loader."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path.cwd() / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------- old PATHS ----------------
# Base directories
DATA_DIR = Path("data")
GENERATED_DIR = DATA_DIR / "generated"

# Input files
PROMPTS_FILE = DATA_DIR / "recording-prompts.json"

# Output Audio Data
RAW_RECORDINGS_DIR = GENERATED_DIR / "raw_recordings"
CLEANED_RECORDINGS_DIR = GENERATED_DIR / "cleaned_recordings" 
PROCESSED_CLIPS_DIR = GENERATED_DIR / "processed_clips"
BALANCED_CLIPS_DIR = GENERATED_DIR / "filtered_balanced_clips"

DIARIZED_RECORDINGS_DIR = GENERATED_DIR / "diarized_recordings"

# Output files
MANIFEST_FILE = GENERATED_DIR / "manifest.csv"
NEW_MANIFEST_FILE = GENERATED_DIR / "new_manifest.csv"
BALANCED_MANIFEST_FILE = GENERATED_DIR / "filtered_balanced_manifest.csv"
FEATURES_FILE = GENERATED_DIR / "vocal_features.csv"

# Model directory
MODEL_DIR = GENERATED_DIR / "model"