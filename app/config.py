"""Configuration loader."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path.cwd() / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Convert all strings containing '/' to Path objects
    for k, v in cfg.items():
        if isinstance(v, str) and "/" in v:
            cfg[k] = Path(v).resolve()  # resolves relative to current working dir
    return cfg

