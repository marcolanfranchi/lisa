# 4-extract-vocal-features.py
import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.progress import track
from config import BALANCED_CLIPS_DIR, FEATURES_FILE

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
N_MFCC = 13           # number of MFCC coefficients (vocal tract shape)
HOP_LENGTH = 512      # hop length for spectral analysis
N_FFT = 2048          # FFT window size
F_MIN = 80            # minimum frequency for vocal analysis
F_MAX = 8000          # maximum frequency for vocal analysis (speech range)
# ----------------------------------------

# setup console
console = Console()

def extract_vocal_tract_features(y, sr):
    """
    Extract MFCC features that represent vocal tract characteristics.
    
    Args:
        y: audio time series
        sr: sample rate
    
    Returns:
        dict of vocal tract features
    """
    # extract MFCCs (skip 0th coefficient which represents energy)
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, 
        n_fft=N_FFT, fmin=F_MIN, fmax=F_MAX
    )[1:]  # skip MFCC_0
    
    # calculate derivatives (capture articulation dynamics)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    features = {}
    
    # statistical features for each MFCC coefficient (skip 0th)
    for i in range(N_MFCC - 1):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_{i+1}_delta_std'] = np.std(mfcc_delta[i])
        features[f'mfcc_{i+1}_delta2_mean'] = np.mean(mfcc_delta2[i])
        features[f'mfcc_{i+1}_delta2_std'] = np.std(mfcc_delta2[i])
    
    return features


def extract_pitch_features(y, sr):
    """
    Extract fundamental frequency features that capture voice pitch characteristics.
    
    Args:
        y: audio time series
        sr: sample rate
    
    Returns:
        dict of pitch features
    """
    features = {}
    
    try:
        # extract F0 using YIN algorithm (more robust for speech)
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, hop_length=HOP_LENGTH)
        
        # remove unvoiced frames (0 Hz)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) > 10:  # need sufficient voiced frames
            # basic pitch statistics
            features['f0_mean'] = np.mean(f0_voiced)
            features['f0_std'] = np.std(f0_voiced)
            features['f0_median'] = np.median(f0_voiced)
            features['f0_range'] = np.max(f0_voiced) - np.min(f0_voiced)
            features['f0_iqr'] = np.percentile(f0_voiced, 75) - np.percentile(f0_voiced, 25)
            
            # pitch dynamics (voice quality indicators)
            f0_diff = np.diff(f0_voiced)
            features['f0_jitter'] = np.std(f0_diff) / np.mean(f0_voiced)  # pitch variation
            features['voicing_rate'] = len(f0_voiced) / len(f0)  # percentage of voiced frames
            
            # pitch contour characteristics
            features['f0_slope'] = np.polyfit(range(len(f0_voiced)), f0_voiced, 1)[0]  # overall trend
            features['f0_contour_std'] = np.std(np.gradient(f0_voiced))  # contour variability
            
        else:
            # insufficient voiced content
            for key in ['f0_mean', 'f0_std', 'f0_median', 'f0_range', 'f0_iqr', 
                       'f0_jitter', 'voicing_rate', 'f0_slope', 'f0_contour_std']:
                features[key] = 0
    
    except Exception as e:
        console.print(f"[yellow]warning: pitch extraction failed, using zeros[/yellow]")
        for key in ['f0_mean', 'f0_std', 'f0_median', 'f0_range', 'f0_iqr', 
                   'f0_jitter', 'voicing_rate', 'f0_slope', 'f0_contour_std']:
            features[key] = 0
    
    return features


def extract_formant_approximation_features(y, sr):
    """
    Extract features that approximate formant characteristics using spectral analysis.
    
    Args:
        y: audio time series
        sr: sample rate
    
    Returns:
        dict of formant-related features
    """
    features = {}
    
    # compute STFT for spectral analysis
    stft = librosa.stft(y, hop_length=HOP_LENGTH, n_fft=N_FFT)
    magnitude = np.abs(stft)
    power = magnitude ** 2
    
    # convert to frequency bins
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    
    # focus on speech frequency range
    speech_mask = (freq_bins >= F_MIN) & (freq_bins <= F_MAX)
    speech_freqs = freq_bins[speech_mask]
    speech_power = power[speech_mask, :]
    
    # avoid division by zero
    speech_power_sum = np.sum(speech_power, axis=0)
    valid_frames = speech_power_sum > 0
    
    if np.any(valid_frames):
        # spectral centroid in speech range (approximates F1/F2 balance)
        spectral_centroid = np.zeros(speech_power.shape[1])
        spectral_centroid[valid_frames] = (
            np.sum(speech_freqs[:, np.newaxis] * speech_power[:, valid_frames], axis=0) / 
            speech_power_sum[valid_frames]
        )
        
        features['speech_spectral_centroid_mean'] = np.mean(spectral_centroid[valid_frames])
        features['speech_spectral_centroid_std'] = np.std(spectral_centroid[valid_frames])
        
        # spectral spread in speech range
        spectral_spread = np.zeros(speech_power.shape[1])
        for i in range(speech_power.shape[1]):
            if valid_frames[i]:
                spectral_spread[i] = np.sqrt(
                    np.sum(((speech_freqs - spectral_centroid[i]) ** 2) * speech_power[:, i]) / 
                    speech_power_sum[i]
                )
        
        features['speech_spectral_spread_mean'] = np.mean(spectral_spread[valid_frames])
        features['speech_spectral_spread_std'] = np.std(spectral_spread[valid_frames])
        
    else:
        features['speech_spectral_centroid_mean'] = 0
        features['speech_spectral_centroid_std'] = 0
        features['speech_spectral_spread_mean'] = 0
        features['speech_spectral_spread_std'] = 0
    
    # approximate formant regions (simplified)
    # F1 region: 300-900 Hz, F2 region: 900-2500 Hz, F3 region: 2500-4000 Hz
    f1_mask = (speech_freqs >= 300) & (speech_freqs <= 900)
    f2_mask = (speech_freqs >= 900) & (speech_freqs <= 2500)
    f3_mask = (speech_freqs >= 2500) & (speech_freqs <= 4000)
    
    if np.any(f1_mask):
        f1_energy = np.mean(np.sum(speech_power[f1_mask, :], axis=0))
        features['f1_energy_mean'] = f1_energy
    else:
        features['f1_energy_mean'] = 0
    
    if np.any(f2_mask):
        f2_energy = np.mean(np.sum(speech_power[f2_mask, :], axis=0))
        features['f2_energy_mean'] = f2_energy
    else:
        features['f2_energy_mean'] = 0
    
    if np.any(f3_mask):
        f3_energy = np.mean(np.sum(speech_power[f3_mask, :], axis=0))
        features['f3_energy_mean'] = f3_energy
    else:
        features['f3_energy_mean'] = 0
    
    # formant ratios (accent indicators)
    if features['f1_energy_mean'] > 0 and features['f2_energy_mean'] > 0:
        features['f2_f1_ratio'] = features['f2_energy_mean'] / features['f1_energy_mean']
    else:
        features['f2_f1_ratio'] = 0
    
    return features


def extract_voice_quality_features(y, sr):
    """
    Extract voice quality features that capture individual vocal characteristics.
    
    Args:
        y: audio time series
        sr: sample rate
    
    Returns:
        dict of voice quality features
    """
    features = {}
    
    # harmonic-to-noise ratio approximation
    # using spectral flatness in speech range
    speech_stft = librosa.stft(y, hop_length=HOP_LENGTH, n_fft=N_FFT)
    speech_magnitude = np.abs(speech_stft)
    
    # focus on speech frequencies
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    speech_mask = (freqs >= F_MIN) & (freqs <= F_MAX)
    speech_mag = speech_magnitude[speech_mask, :]
    
    # spectral flatness (measure of harmonicity)
    geometric_mean = np.exp(np.mean(np.log(speech_mag + 1e-8), axis=0))
    arithmetic_mean = np.mean(speech_mag, axis=0)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
    
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    
    # zero crossing rate (voicing indicator)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # spectral rolloff in speech range
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=HOP_LENGTH, roll_percent=0.85
    )[0]
    
    # normalize by maximum speech frequency
    normalized_rolloff = spectral_rolloff / F_MAX
    features['spectral_rolloff_mean'] = np.mean(normalized_rolloff)
    features['spectral_rolloff_std'] = np.std(normalized_rolloff)
    
    return features


def extract_vocal_features(file_path):
    """
    Extract only vocal characteristics from an audio file.
    
    Args:
        file_path: pathlib.Path to audio file
    
    Returns:
        dict with vocal features or None if error
    """
    try:
        # load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # pre-emphasize to enhance higher frequencies (standard for speech)
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # extract only vocal characteristics
        features = {}
        features.update(extract_vocal_tract_features(y, sr))
        features.update(extract_pitch_features(y, sr))
        features.update(extract_formant_approximation_features(y, sr))
        features.update(extract_voice_quality_features(y, sr))
        
        # add metadata
        features['speaker_id'] = file_path.parent.name
        features['clip_filename'] = file_path.name
        features['duration'] = len(y) / sr
        
        return features
        
    except Exception as e:
        console.print(f"[red]error extracting vocal features from {file_path.name}: {e}[/red]")
        return None


def main():
    """
    Main function to extract vocal features from all balanced clips.
    
    Args:
        none
    
    Returns:
        saves vocal feature matrix to CSV file

    Pipeline Step:
        4/5
    
    Expects:
        balanced clips in BALANCED_CLIPS_DIR
    """
    
    if not BALANCED_CLIPS_DIR.exists():
        console.print(f"[red]error: balanced clips directory not found: {BALANCED_CLIPS_DIR}[/red]")
        return
    
    # find all speaker directories
    speaker_dirs = [d for d in BALANCED_CLIPS_DIR.iterdir() if d.is_dir()]
    
    if not speaker_dirs:
        console.print(f"[red]error: no speaker directories found in {BALANCED_CLIPS_DIR}[/red]")
        return
    
    console.rule("[bold green]starting vocal feature extraction[/bold green]")
    console.print(f"[cyan]found {len(speaker_dirs)} speaker(s) to process[/cyan]")
    console.print(f"[cyan]sample rate: {SAMPLE_RATE} Hz[/cyan]")
    console.print(f"[cyan]speech frequency range: {F_MIN}-{F_MAX} Hz[/cyan]")
    console.print(f"[cyan]MFCC coefficients: {N_MFCC-1} (excluding energy)[/cyan]")
    
    all_features = []
    total_clips = 0
    processed_clips = 0
    
    # process each speaker
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        console.rule(f"[bold green]processing speaker: {speaker_id}[/bold green]")
        
        # find all wav files for this speaker
        wav_files = list(speaker_dir.glob("*.wav"))
        
        if not wav_files:
            console.print(f"[yellow]no .wav files found for {speaker_id}[/yellow]")
            continue
        
        console.print(f"[cyan]extracting vocal features from {len(wav_files)} clips for {speaker_id}[/cyan]")
        
        # extract features from each clip
        for wav_file in track(wav_files, description=f"processing {speaker_id} clips"):
            total_clips += 1
            
            features = extract_vocal_features(wav_file)
            if features:
                all_features.append(features)
                processed_clips += 1
    
    # save features to CSV
    if all_features:
        console.rule("[bold cyan]saving vocal feature matrix[/bold cyan]")
        console.print(f"[cyan]creating feature matrix with {len(all_features)} samples[/cyan]")
        
        # create DataFrame
        df = pd.DataFrame(all_features)
        
        # ensure output directory exists
        FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # save to CSV
        df.to_csv(FEATURES_FILE, index=False)
        
        console.print(f"[bold green]vocal feature matrix saved: {FEATURES_FILE}[/bold green]")
        console.print(f"[bold green]matrix dimensions: {df.shape[0]} samples x {df.shape[1]} features[/bold green]")
        
        # show feature summary
        console.print(f"[cyan]speakers: {df['speaker_id'].unique().tolist()}[/cyan]")
        console.print(f"[cyan]clips per speaker: {df['speaker_id'].value_counts().to_dict()}[/cyan]")
        
        # feature breakdown
        vocal_tract_features = len([col for col in df.columns if col.startswith('mfcc_')])
        pitch_features = len([col for col in df.columns if col.startswith('f0_') or 'voicing' in col])
        formant_features = len([col for col in df.columns if any(x in col for x in ['f1_', 'f2_', 'f3_', 'speech_spectral'])])
        quality_features = len([col for col in df.columns if any(x in col for x in ['spectral_flatness', 'zcr_', 'spectral_rolloff'])])
        
        console.print(f"[green]feature breakdown:[/green]")
        console.print(f"  vocal tract (MFCC): {vocal_tract_features}")
        console.print(f"  pitch/prosody: {pitch_features}")
        console.print(f"  formant-related: {formant_features}")
        console.print(f"  voice quality: {quality_features}")
        
    else:
        console.print(f"[red]no features extracted, CSV not created[/red]")
    
    # final summary
    console.rule("[bold green]vocal feature extraction complete![/bold green]")
    console.print(f"[bold green]processed {processed_clips}/{total_clips} clips successfully[/bold green]")
    console.print(f"[yellow]features focus on vocal characteristics only[/yellow]")
    console.print(f"[yellow]removed environment-sensitive features (tempo, silence, energy levels)[/yellow]")
    
    if processed_clips < total_clips:
        console.print(f"[yellow]warning: {total_clips - processed_clips} clips had errors[/yellow]")


if __name__ == "__main__":
    main()