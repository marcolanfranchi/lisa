import streamlit as st
import pathlib
import os
import sys
import numpy as np
import librosa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.config import RAW_RECORDINGS_DIR, PROCESSED_CLIPS_DIR, CLEANED_RECORDINGS_DIR, BALANCED_CLIPS_DIR

CACHE_TTL = 60*10  # seconds

# Speaker colours for visualizations
SPEAKER_COLOURS = {
    "marco": "#865bae",   # Purple
    "georgii": "#f0ad72",   # Orange
    "vova": "#5D825D",     # Green
    "kolya": "#2668ca",   # Blue
}

def speaker_name(speaker_id):
    """ returns the formatted (colour-highlighted) speaker name """
    colour = SPEAKER_COLOURS.get(speaker_id, "#8A8989")  # Default to grey if not found
    return f"<span style='color:{colour}; font-weight:bold'>{speaker_id}</span>"


# TODO: remove this function and its references and use load_audio_data (below) instead
@st.cache_data(show_spinner=False, ttl=60)
def get_audio_metadata(from_step=0):
    """Extract metadata from the audio file path."""

    rec_length_min = 1
    rec_per_speaker = 5
    split_length_sec = 2
    split_overlap = 0.5 * 100

    # get # of folders in the path
    path = pathlib.Path(RAW_RECORDINGS_DIR)
    n_speakers = 0
    for folder in path.iterdir():
        if folder.is_dir():
            n_speakers += 1

    # get num of recordings (check each folder)
    n_recordings = 0
    for folder in path.iterdir():
        if folder.is_dir():
            n_recordings += len(list(folder.glob("*.wav")))

    if from_step <= 1:
        split_length_sec = "0"
        split_overlap = "0"
        n_clips = "N/A"
        filtered_clips = "N/A"
    else:
        n_clips = 0
        clips_path = pathlib.Path(PROCESSED_CLIPS_DIR)
        for folder in clips_path.iterdir():
            if folder.is_dir():
                n_clips += len(list(folder.glob("*.wav")))

        # estimate filtered clips (assume 20% are silent)
        filtered_clips = int(n_clips * 0.8)
    
    raw_data_counts = {
        "num_speakers": n_speakers,
        "num_recordings": n_recordings,
        "rec_length_min": rec_length_min,
        "total_rec_length": n_speakers * rec_per_speaker * rec_length_min,
        "rec_per_speaker": rec_per_speaker,
        "split_length_sec": split_length_sec,
        "split_overlap": split_overlap,
        "num_clips": n_clips,
        "filtered_clips": filtered_clips
    }
    return raw_data_counts


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_audio_data(data_path):
    """Load audio files and extract metadata - shared between steps"""
    audio_data = {} 
    
    if not os.path.exists(data_path):
        return audio_data
    
    for speaker_folder in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker_folder)
        if os.path.isdir(speaker_path):
            audio_data[speaker_folder] = {}
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(speaker_path, audio_file)
                    try:
                        y, sr = librosa.load(file_path, sr=None)
                        duration = len(y) / sr
                        
                        rms = librosa.feature.rms(y=y)[0]
                        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                        
                        audio_data[speaker_folder][audio_file] = {
                            'path': file_path,
                            'duration': duration,
                            'sample_rate': sr,
                            'samples': len(y),
                            'rms_mean': np.mean(rms),
                            'rms_std': np.std(rms),
                            'spectral_centroid_mean': np.mean(spectral_centroids),
                            'spectral_centroid_std': np.std(spectral_centroids),
                            'max_amplitude': np.max(np.abs(y)),
                            'audio_data': y
                        }
                    except Exception as e:
                        st.error(f"Error loading {file_path}: {str(e)}")
    
    return audio_data