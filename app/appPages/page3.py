import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import librosa
import plotly.express as px
import plotly.graph_objects as go
from streamlit_advanced_audio import audix, WaveSurferOptions
from appPages.components import section_header, blank_lines
from utils import SPEAKER_COLOURS, load_audio_data
from src.config import BALANCED_CLIPS_DIR, BALANCED_MANIFEST_FILE


def page3():
    """Display the page for step 3-filter-and-balance.py."""
    # ==============================================================================
    # Header
    # ==============================================================================
    section_header(
        "Filtered & Balanced Dataset",
        "Final stage of preprocessing. Filters out low-quality clips based on RMS energy and balances the dataset across speakers. "
        "The balanced clips are stored in `data/generated/filtered_balanced_clips`."
    )

    # ==============================================================================
    # Load Balanced Data
    # ==============================================================================
    st.write(BALANCED_CLIPS_DIR)
    audio_data = load_audio_data(BALANCED_CLIPS_DIR)
    manifest_df = pd.read_csv(BALANCED_MANIFEST_FILE) if os.path.exists(BALANCED_MANIFEST_FILE) else None

    if os.path.exists(BALANCED_MANIFEST_FILE):
        try:
            manifest_df = pd.read_csv(BALANCED_MANIFEST_FILE)
        except Exception as e:
            st.error(f"Error loading manifest: {str(e)}")

    if not audio_data:
        st.warning(f"No balanced audio data found in `{BALANCED_CLIPS_DIR}`. Please run 3-filter-and-balance.py.")
        return

    # ==============================================================================
    # Summary Metrics
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Balanced Dataset Summary")

    total_speakers = len(audio_data)
    total_clips = sum(len(clips) for clips in audio_data.values())
    first_clip = next(iter(next(iter(audio_data.values())).values()))
    clip_duration = first_clip.get("duration", 0)
    total_audio_length = (clip_duration * total_clips) / 60

    with st.container(border=False):
        subBorders = True
        metricHeight = 120
        cols = st.columns(4)
        with cols[0]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(":small[Speakers:]")
                st.markdown(f"#### {total_speakers}")
        with cols[1]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(":small[Clips per speaker:]")
                st.markdown(f"#### {len(next(iter(audio_data.values())))}")
        with cols[2]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(":small[Total clips:]")
                st.markdown(f"#### {total_clips}")
        with cols[3]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(":small[Total length:]")
                st.markdown(f"#### {total_audio_length:.1f} min")

    # ==============================================================================
    # Balance Visualization
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Data Balance Verification")

    clip_counts = {spk: len(clips) for spk, clips in audio_data.items()}
    balance_df = pd.DataFrame(list(clip_counts.items()), columns=["Speaker", "Clip Count"])

    fig = px.bar(
        balance_df,
        x="Speaker",
        y="Clip Count",
        color="Speaker",
        color_discrete_map=SPEAKER_COLOURS,
        title="Number of Balanced Clips per Speaker",
        text="Clip Count"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig)

    min_clips = balance_df["Clip Count"].min()
    max_clips = balance_df["Clip Count"].max()

    if min_clips == max_clips:
        st.success(f"✅ Perfect balance achieved — {min_clips} clips per speaker.")
    else:
        st.warning(f"⚠️ Dataset not perfectly balanced ({min_clips}–{max_clips} clips per speaker).")

    # ==============================================================================
    # RMS Distribution
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### RMS Energy Distribution (Post-Filtering)")

    rms_data = []
    for speaker, clips in audio_data.items():
        for clip_name, data in clips.items():
            rms_data.append({
                "Speaker": speaker,
                "Clip": clip_name,
                "RMS Mean": data["rms_mean"]
            })

    rms_df = pd.DataFrame(rms_data)
    rms_fig = px.box(
        rms_df,
        x="Speaker",
        y="RMS Mean",
        color="Speaker",
        color_discrete_map=SPEAKER_COLOURS,
        title="RMS Energy per Speaker (Filtered & Balanced)"
    )
    st.plotly_chart(rms_fig)

    # ==============================================================================
    # Listen to Random Balanced Clips
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Sample Balanced Clips")

    all_speakers = list(audio_data.keys())
    selected_speaker = st.selectbox("Select speaker:", all_speakers)
    speaker_clips = list(audio_data[selected_speaker].items())

    # Pick a random clip (or allow manual selection)
    clip_names = [clip for clip, _ in speaker_clips]
    selected_clip = st.selectbox("Select clip:", clip_names, index=random.randint(0, len(clip_names) - 1))
    clip_data = audio_data[selected_speaker][selected_clip]

    st.write(
        f":blue-badge[{selected_speaker}] "
        f":orange-badge[RMS: {clip_data['rms_mean']:.4f}] "
        f":gray-badge[Duration: {clip_data['duration']:.2f}s]"
    )

    # Audio player
    options = WaveSurferOptions(
        wave_color="#4682B4",
        progress_color="#2E8B57",
        height=80,
        bar_width=2,
        bar_gap=1,
        normalize=True
    )

    try:
        audix(clip_data["path"], wavesurfer_options=options)
    except Exception as e:
        st.error(f"Error loading audio player: {str(e)}")
        st.audio(clip_data["path"])

    # ==============================================================================
    # Manifest Table
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Balanced Dataset Manifest")

    if manifest_df is not None and not manifest_df.empty:
        st.dataframe(manifest_df, hide_index=True, use_container_width=True)
    else:
        st.info("No manifest file found or manifest is empty.")

