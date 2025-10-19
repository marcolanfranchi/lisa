from appPages.components import section_header
from appPages.reusablePages.audioAnalysis import show_audio_analysis_page

def page1():
    """Display the page for step 1-clean-audio.py."""

    # ==============================================================================
    # Header
    # ==============================================================================
    section_header("Audio Cleaning", "Data cleaning phase. Cleans and preprocesses the raw audio recordings and stores them in \
                    `data/generated/cleaned_recordings`.")
    
    # ==============================================================================
    # Reusable Audio Analysis Page (on cleaned data)
    # ==============================================================================
    show_audio_analysis_page(
        version="cleaned",
        wave_color="#6A5ACD",
        progress_color="#483D8B"
    )
