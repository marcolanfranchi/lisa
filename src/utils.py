from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple

logger = logging.getLogger(__name__)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_audio(path: Path | str, sr: int) -> Tuple[np.ndarray, int]:
    """Load audio; returns mono float32 signal and sample rate."""
    data, file_sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if file_sr != sr:
        data = librosa.resample(data.astype(float), orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return data.astype(np.float32), file_sr


def write_audio(path: Path | str, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, samplerate=sr)


def rms_energy(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))
