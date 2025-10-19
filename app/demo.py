# demo.py
# Gradio Live Demo of the model (with text to speech as well)
# -------------------------------------------------------------------------
# When this file is running at the same time as the Streamlit app, 
# the app will display this Gradio demo in an iframe on the "Home" page. 
# -------------------------------------------------------------------------
import gradio as gr
import speech_recognition as sr
import numpy as np
import io
import wave
from collections import deque
import time

recognizer = sr.Recognizer()

# For text smoothing
recent_transcript = deque(maxlen=4)
last_update_time = 0
pause_threshold = 2.0

# For audio buffering
audio_buffer = []
buffer_duration = 3  # seconds
last_buffer_time = time.time()


def transcribe_audio(audio):
    global audio_buffer, last_buffer_time
    global recent_transcript, last_update_time

    if audio is None:
        return "Listening..."

    try:
        sample_rate, audio_data = audio

        # Normalize dtype
        if audio_data.dtype in (np.float32, np.float64):
            audio_data = (audio_data * 32767).astype(np.int16)

        # Mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(np.int16)

        # Append to buffer
        audio_buffer.append(audio_data)
        now = time.time()

        # Only run recognition if buffer "full enough"
        if now - last_buffer_time < buffer_duration:
            return " ".join(recent_transcript) if recent_transcript else "Listening..."

        # Stitch buffer
        full_audio = np.concatenate(audio_buffer)
        audio_buffer = []  # reset buffer
        last_buffer_time = now

        # Write wav
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(full_audio.tobytes())
        wav_io.seek(0)

        # Recognize
        with sr.AudioFile(wav_io) as source:
            audio_chunk = recognizer.record(source)
            text = recognizer.recognize_google(audio_chunk, language="en-US").strip()

        # Smooth transcript flow
        if text:
            if now - last_update_time > pause_threshold:
                recent_transcript.clear()
            recent_transcript.append(text)
            last_update_time = now

        return " ".join(recent_transcript) if recent_transcript else "Listening..."

    except sr.UnknownValueError:
        return " ".join(recent_transcript) if recent_transcript else "Listening..."
    except Exception as e:
        return f"[error: {e}]"


# Create Gradio interface with custom CSS
css = """
.gradio-container {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 18px !important;
    height: 100vh !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

/* Remove borders + shadows globally */
.block, .form, .input-container, .show_textbox_border, textarea {
    background: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 18px !important;
    color: #000000 !important;
    width: 700px !important;
    max-width: 700px !important;
}

/* Buttons (mic, controls, etc.) */
.mic-wrap {
    background: #ffffff !important;
    color: #000000 !important;
    box-shadow: none !important;
    font-size: 16px !important;
    width: 100% !important;
    max-width: 700px !important;
    margin: 0 auto !important;
}

.form.svelte-1vd8eap {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    style: none !important;
}

button {
    background: #f0f0f0 !important;
    color: #000000 !important;
    border: 1px solid #ccc !important;
    box-shadow: none !important;
    font-size: 16px !important;
    border-radius: 6px !important;
}

select {
    background: #f0f0f0 !important;
    color: #000000 !important;
    border: 1px solid #ccc !important;
    box-shadow: none !important;
    font-size: 16px !important;
    border-radius: 6px !important;
}

button:hover {
    background: #e0e0e0 !important;
    cursor: pointer;
}

footer, .footer {
    display: none !important;
}

.icon-button-wrapper.top-panel.hide-top-corner.svelte-ud4hud {
    display: none !important;
}

.button-wrap.svelte-10cpz3p {
    display: none !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Default(
        font="serif"
    ),
    css=css,
    title="Live Speech Demo",

) as demo:
    
    # Webcam display (just for visual feedback)
    webcam = gr.Image(
        sources=["webcam"],
        streaming=True,
        show_label=False,
        show_download_button=False,
        mirror_webcam=True,
        height=400,
        width=600,
        watermark='srm-01'
    )
    
    # Audio input
    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        label=None,
        show_label=False,
        show_download_button=False,
    )
    
    # Transcription output
    transcription_output = gr.Textbox(
        label="",
        placeholder="Live captions will appear here...",
        lines=8,
        max_lines=15,
        show_label=False
    )
    
    # Real-time transcription with streaming
    audio_input.stream(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=transcription_output,
        stream_every=1.5
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )