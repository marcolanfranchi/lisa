# src/4-extract-features.py
import librosa
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from config import load_config

console = Console()

cfg = load_config() 

def extract_basic_mfcc_features(y, sr):
    """Extract mean and std of MFCCs and their deltas."""
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=cfg["N_MFCC"], hop_length=cfg["HOP_LENGTH"], n_fft=cfg["N_FFT"]
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    
    features = {}
    for i in range(cfg["N_MFCC"]):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfccs[i])
        features[f"mfcc_{i+1}_delta_mean"] = np.mean(mfcc_delta[i])
        features[f"mfcc_{i+1}_delta_std"] = np.std(mfcc_delta[i])
    
    return features

def extract_features(file_path):
    """Extract core vocal features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=cfg["SAMPLE_RATE"])
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
    if not cfg["BALANCED_CLIPS_DIR"].exists():
        console.print(f'[red]Error: Missing directory {cfg["BALANCED_CLIPS_DIR"]}[/red]')
        return

    speaker_dirs = [d for d in cfg["BALANCED_CLIPS_DIR"].iterdir() if d.is_dir()]
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
    cfg["FEATURES_FILE"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["FEATURES_FILE"], index=False)

    console.print("\n")
    console.print("[bold green]Sample of extracted features:[/bold green]")
    console.print("\n")
    sample_cols = list(df.columns[:5]) + ["speaker_id", "clip_filename", "duration"]
    console.print(df[sample_cols].head(8))
    console.print("\n")

    console.print(f'[green]Saved {len(df)} samples with {df.shape[1]} features to {cfg["FEATURES_FILE"]}[/green]')

if __name__ == "__main__":
    main()
