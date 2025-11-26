import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from appPages.components import blank_lines, audio_player_component
from streamlit_advanced_audio import audix, WaveSurferOptions
from utils import SPEAKER_COLOURS, load_audio_data
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import load_config

cfg = load_config()

def create_waveform_comparison(audio_data, selected_speakers=None, selected_recording="A.wav"):
    """Create overlaid waveform visualization for selected speakers."""
    if not audio_data or not selected_speakers:
        return None

    fig = go.Figure()

    for speaker in selected_speakers:
        if speaker in audio_data:
            y = audio_data[speaker][selected_recording]['audio_data']
            sr = audio_data[speaker][selected_recording]['sample_rate']
            time = np.linspace(0, len(y) / sr, len(y))
            
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=y,
                    mode='lines',
                    name=speaker,
                    line=dict(color=SPEAKER_COLOURS.get(speaker, "#8A8989"), width=1),
                    opacity=0.7
                )
            )

    fig.update_layout(
        title=f"Audio Waveforms – {selected_recording}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=400,
        legend=dict(
            title="Speakers",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=60, b=40)
    )

    fig.update_yaxes(range=[-1, 1])

    return fig


def create_audio_statistics_chart(audio_data, chart_type="rms"):
    """Create audio statistics comparison charts"""
    if not audio_data:
        return None
    
    # Prepare data for plotting
    plot_data = []
    for speaker, recordings in audio_data.items():
        for recording, data in recordings.items():
            plot_data.append({
                'Speaker': speaker,
                'Recording': recording,
                'RMS Mean': data['rms_mean'],
                'RMS Std': data['rms_std'],
                'Spectral Centroid Mean': data['spectral_centroid_mean'],
                'Spectral Centroid Std': data['spectral_centroid_std'],
                'Max Amplitude': data['max_amplitude'],
                'Duration': data['duration']
            })
    
    df = pd.DataFrame(plot_data)
    
    color_map = {speaker: SPEAKER_COLOURS.get(speaker, "#8A8989") for speaker in df['Speaker'].unique()}

    blank_lines(1)

    if chart_type == "rms":
        st.write("**Mean RMS (root mean square) energy** in audio samples is the average loudness of a signal over a period of time, \
                 calculated by squaring the amplitude of each sample, averaging the squares, and then taking the square root. \
                 A higher RMS value indicates greater average power and loudness of the audio signal.")
        fig = px.box(
            df, x='Speaker', y='RMS Mean',
            title="RMS Energy Distribution by Speaker",
            color='Speaker',
            color_discrete_map=color_map
        )
        # Fix y-axis range
        fig.update_yaxes(range=[0, 0.05])
        
    elif chart_type == "spectral":
        st.write("The **mean spectral centroid** in audio samples is a measure of the 'center of gravity' of a signal's \
                 magnitude spectrum, indicating the average frequency where most of its energy is concentrated. It is \
                 calculated as a weighted mean of the frequencies, with the magnitudes of the spectrum at each frequency \
                 serving as the weights. A higher spectral centroid corresponds to a 'brighter' sound, while a lower \
                 centroid indicates a 'darker' sound")
        fig = px.box(
            df, x='Speaker', y='Spectral Centroid Mean',
            title="Spectral Centroid Distribution by Speaker",
            color='Speaker',
            color_discrete_map=color_map
        )
    elif chart_type == "amplitude":
        st.write("The **maximum amplitude** in audio samples represents the loudness or volume of a sound at its loudest \
                 point. A higher maximum amplitude indicates a louder sound, while a lower maximum amplitude indicates \
                 a softer sound. In a digital audio file, this value is the peak displacement of the sound wave from its \
                 resting position.")
        fig = px.box(
            df, x='Speaker', y='Max Amplitude',
            title="Maximum Amplitude Distribution by Speaker",
            color='Speaker',
            color_discrete_map=color_map
        )
    
    return fig


def show_audio_analysis_page(
    version: str = "raw",  # 'raw' or 'cleaned'
    wave_color="#A1A1A1",
    progress_color="#3a3a3a"
):
    """
    Shared audio analysis page structure.
    version: 'raw' or 'cleaned'
    config: a module or dict containing paths, e.g. config.RAW_AUDIO_PATH / config.CLEANED_AUDIO_PATH
    """

    # Select path + text labels
    if version not in ("raw", "cleaned"):
        raise ValueError("version must be 'raw' or 'cleaned'")

    # Get data path from config
    if version == "raw":
        data_path = cfg["RAW_RECORDINGS_DIR"]
    elif version == "cleaned":
        data_path = cfg["CLEANED_RECORDINGS_DIR"]

    # --- UI labels ---
    version_label = "Raw" if version == "raw" else "Cleaned"
    section_title = f"#### {version_label} Data Summary"

    # --- Load data ---
    blank_lines(2)
    st.markdown(section_title)
    audio_data = load_audio_data(data_path)

    if not audio_data:
        st.warning(f"No {version_label.lower()} audio data found in {data_path}")
        return

    first_recording = next(iter(next(iter(audio_data.values())).values()))
    duration_min = first_recording.get("duration", 0) / 60
    n_recordings = sum(len(recs) for recs in audio_data.values())

    # --- Summary metrics ---
    with st.container(border=False):
        subBorders = True
        metricHeight = 120
        r1columns = st.columns(4)
        with r1columns[0]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(":small[Speakers:]")
                st.markdown(f"#### {len(audio_data)}")
        with r1columns[1]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[{version_label} rec. length:]")
                st.markdown(f"#### {duration_min:.1f} min")
        with r1columns[2]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(":small[Recordings:]")
                st.markdown(f"#### {n_recordings}")
        with r1columns[3]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Total rec. length:]")
                st.markdown(f"#### {duration_min * n_recordings:.1f} min")

    # --- Audio file player section ---
    blank_lines(2)
    st.write("#### Audio Recordings")
    
    # Display each player row
    audio_player_component(audio_data, wave_color=wave_color, progress_color=progress_color)

    speakers = list(audio_data.keys())
    unique_recordings = sorted({rec for recs in audio_data.values() for rec in recs})

    # --- Waveform comparison ---
    blank_lines(2)
    st.markdown("#### Waveform Comparison")

    filter_cols = st.columns(2)
    with filter_cols[0]:
        selected_speakers = st.multiselect(
            "Select speakers to compare:",
            speakers,
            default=speakers,
        )
    with filter_cols[1]:
        selected_recording = st.selectbox(
            "Select recording for comparison:",
            unique_recordings,
            index=0,
        )

    if selected_speakers:
        waveform_fig = create_waveform_comparison(audio_data, selected_speakers, selected_recording)
        if waveform_fig:
            st.plotly_chart(waveform_fig)

    # --- Audio statistics charts ---
    blank_lines(2)
    st.markdown("#### Audio Graphs")

    chart_type = st.selectbox(
        "Select analysis type:",
        ["rms", "spectral", "amplitude"],
        format_func=lambda x: {
            "rms": "RMS Energy",
            "spectral": "Spectral Centroid",
            "amplitude": "Max Amplitude",
        }[x],
    )

    stats_fig = create_audio_statistics_chart(audio_data, chart_type)
    if stats_fig:
        st.plotly_chart(stats_fig)

    # --- Recording details table ---
    blank_lines(2)
    st.markdown("#### Recording Details")

    summary_data = []
    for speaker, recordings in audio_data.items():
        for recording, data in recordings.items():
            summary_data.append({
                "Speaker": speaker,
                "Recording": recording,
                "Duration (s)": f"{data['duration']:.2f}",
                "Sample Rate": data["sample_rate"],
                "RMS Energy": f"{data['rms_mean']:.4f}",
                "Spectral Centroid": f"{data['spectral_centroid_mean']:.2f}",
                "Max Amplitude": f"{data['max_amplitude']:.4f}",
            })

    st.dataframe(pd.DataFrame(summary_data), hide_index=True)