import streamlit as st
from streamlit_advanced_audio import audix, WaveSurferOptions
from utils import get_audio_metadata

# ==============================================================================
# Headers/Text components
# ==============================================================================

def section_header(title: str, description: str) -> None:
    """Render a section header."""
    st.header(f"*LISA: {title}*")
    st.markdown(f":small[{description}]")

# ==============================================================================
# Metrics components
# ==============================================================================

def data_metrics(from_step=0) -> None:
    """Display data metrics placeholder."""

    data = get_audio_metadata(from_step=from_step)

    st.markdown("#### Curent Data")

    with st.container(border=True):
        subBorders = False
        metricHeight = 90
        r1columns = st.columns(4)
        with r1columns[0]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(":small[Speakers:]")
                st.markdown(f"#### {data.get('num_speakers', 'N/A')}")
        with r1columns[2]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Raw rec. length:]")
                st.markdown(f"#### {data.get('rec_length_min', '0')}min")
        with r1columns[1]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Recordings:]") # :small[(**{data.get('recs_per_speaker', 'N/A')}** recs./speaker)]
                st.markdown(f"#### {data.get('num_recordings', 'N/A')}")
        with r1columns[3]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Total rec. length:]") # :small[(**{data.get('total_rec_length', 'N/A')}**min/speaker)]
                st.markdown(f"#### {data.get('total_rec_length', '0')}min")

        r2columns = st.columns(4)
        with r2columns[0]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Split clip length:]")
                st.markdown(f"#### {data.get('split_length_sec', '0')}sec")
        with r2columns[1]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Split overlap:]")
                st.markdown(f"#### {data.get('split_overlap', '0')}%")
        with r2columns[2]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Split clips:]") # :small[(**{data.get('split_length_sec', '0')}sec** clips with **{data.get('split_overlap', '0')}%** overlap)]
                st.markdown(f"#### {data.get('num_clips', 'N/A')}")
        with r2columns[3]:
            with st.container(border=subBorders, height=metricHeight):
                st.markdown(f":small[Filtered clips:]") # :small[(**0%** silence + balanced)]
                st.markdown(f"#### {data.get('filtered_clips', 'N/A')}")

# ==============================================================================
# Audio components
# ==============================================================================

def audio_player_component(
    audio_data: dict,
    wave_color="#A1A1A1",
    progress_color="#3a3a3a",
):
    """
    A reusable Streamlit component for selecting and playing one audio file.
    """

    speakers = list(audio_data.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_speaker = st.selectbox(
            f"Speaker ",
            speakers,
            key=f"speaker_",
        )

    with col2:
        recordings = sorted(audio_data[selected_speaker].keys()) if selected_speaker else []
        selected_recording = st.selectbox(
            f"Recording",
            recordings,
            key=f"recording_",
        )

    if selected_speaker and selected_recording:
        data = audio_data[selected_speaker][selected_recording]

        st.markdown(
            f"**{selected_recording}**  "
            f":blue-badge[Duration: {data['duration']:.2f}s] "
            f":orange-badge[Sample Rate: {data['sample_rate']}Hz]"
        )

        options = WaveSurferOptions(
            wave_color=wave_color,
            progress_color=progress_color,
            height=60,
            bar_width=2,
            bar_gap=1,
            normalize=False,
        )

        try:
            audix(data["path"], wavesurfer_options=options, key=f"audix_")
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            st.audio(data["path"])


def blank_lines(n=1):
    """Insert blank lines for vertical spacing. Default is 1 line."""
    for _ in range(n):
        st.write("")
