from appPages.components import section_header
from appPages.reusablePages.audioAnalysis import show_audio_analysis_page

def page0():
    """Page for Step 0: Data Collection and Diarization"""

    # ==============================================================================
    # Header
    # ==============================================================================
    section_header("Raw Data Collection", "Data collection phase. Gathers audio from a provided file or YouTube link, \
                  performs speaker diarization to identify and segment different speakers, and saves the \
                  processed audio clips in `data/generated/raw_recordings/<audio_id>/`.")

    # ==============================================================================
    # Reusable Audio Analysis Page (on raw data)
    # ==============================================================================
    show_audio_analysis_page(
        version="raw",
        wave_color="#FF6F61",
        progress_color="#B33A3A"
    )
