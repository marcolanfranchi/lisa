# src/4-extract-features-simple.py
from .config import BALANCED_CLIPS_DIR, FEATURES_FILE
import librosa
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
# ----------------------------------------

console = Console()

def extract_basic_mfcc_features(y, sr):
    """Extract mean and std of MFCCs and their deltas."""
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    
    features = {}
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfccs[i])
        features[f"mfcc_{i+1}_delta_mean"] = np.mean(mfcc_delta[i])
        features[f"mfcc_{i+1}_delta_std"] = np.std(mfcc_delta[i])
    
    return features

def extract_features(file_path):
    """Extract core vocal features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = librosa.effects.preemphasis(y, coef=0.97)

        features = extract_basic_mfcc_features(y, sr)
        features["speaker_id"] = file_path.parent.name
        features["clip_filename"] = file_path.name
        features["duration"] = len(y) / sr
        return features

    except Exception as e:
        console.print(f"[red]Error extracting features from {file_path.name}: {e}[/red]")
        return None

def main():
    if not BALANCED_CLIPS_DIR.exists():
        console.print(f"[red]Error: Missing directory {BALANCED_CLIPS_DIR}[/red]")
        return

    speaker_dirs = [d for d in BALANCED_CLIPS_DIR.iterdir() if d.is_dir()]
    all_features = []

    for speaker_dir in speaker_dirs:
        wav_files = list(speaker_dir.glob("*.wav"))
        if not wav_files:
            continue

        console.print(f"[cyan]Processing {len(wav_files)} clips for {speaker_dir.name}[/cyan]")
        for wav_file in track(wav_files, description=f"{speaker_dir.name}"):
            f = extract_features(wav_file)
            if f:
                all_features.append(f)

    if not all_features:
        console.print("[red]No features extracted![/red]")
        return

    df = pd.DataFrame(all_features)
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)

    console.print(f"[green]Saved {len(df)} samples with {df.shape[1]} features to {FEATURES_FILE}[/green]")

if __name__ == "__main__":
    main()
