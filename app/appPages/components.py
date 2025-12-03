import streamlit as st
from streamlit_advanced_audio import audix, WaveSurferOptions


def section_header(title: str, description: str) -> None:
    """Render a section header."""
    st.header(f"*lisa: {title}*")
    st.markdown(f":small[{description}]")


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
    """Insert blank lines for vertical spacing in Streamlit. Default is 1 line."""
    for _ in range(n):
        st.write("")
