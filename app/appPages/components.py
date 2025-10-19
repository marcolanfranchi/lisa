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

def audioPlayer(
    file_path: str,
    wave_color: str = "#A1A1A1",
    progress_color: str = "#3a3a3a",
    height: int = 60,
    normalize: bool = False,
    ) -> None:
    pass
    
def render_audio_player(
    file_path: str,
    wave_color: str = "#A1A1A1",
    progress_color: str = "#3a3a3a",
    height: int = 60,
    normalize: bool = False,
    ) -> None:
    
    """Render audio player with wavesurfer."""
    options = WaveSurferOptions(
        wave_color=wave_color,
        progress_color=progress_color,
        height=height,
        bar_width=2,
        bar_gap=1,
        normalize=normalize,
    )
    
    try:
        audix(file_path, wavesurfer_options=options)
    except Exception as e:
        st.error(f"Error loading audio player: {str(e)}")
        st.audio(file_path)


def blank_lines(n=1):
    """Insert blank lines for vertical spacing. Default is 1 line."""
    for _ in range(n):
        st.write("")
