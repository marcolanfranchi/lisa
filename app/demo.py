# demo.py
# -------------------------------------------------------------------------
# Gradio Live Demo of combined Speech Recognition + Speaker ID
# -------------------------------------------------------------------------

import gradio as gr
import speech_recognition as sr
import numpy as np
import io
import os
import sys
import wave
import time
import pickle
from collections import deque, Counter
from pathlib import Path
import importlib
import threading
from queue import Queue
from config import load_config

# === Import your feature extraction + cleaning utilities ===
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
step1 = importlib.import_module("src.1-clean-audio")
step4 = importlib.import_module("src.4-extract-features")
cfg = load_config()

# === Load trained model and scaler ===
with open(cfg["MODEL_DIR"] / "speaker_recognition_knn.pkl", "rb") as f:
    model = pickle.load(f)
with open(cfg["MODEL_DIR"] / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Speech recognizer ===
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# === Global state class to avoid variable issues ===
class AudioProcessingState:
    def __init__(self):
        self.buffer = None
        self.sample_rate = None
        self.recent_transcript = deque(maxlen=4)
        self.last_update_time = 0
        self.pause_threshold = 2.0
        self.last_process_time = time.time()
        self.recent_predictions = deque(maxlen=5)
        self.current_speaker = "Detecting..."
        self.current_confidence = 0.0
        self.processing_times = deque(maxlen=10)
        self.prediction_queue = Queue()

# Create global state instance
state = AudioProcessingState()

# === Configuration ===
TARGET_SAMPLE_RATE = 16000
CLIP_LENGTH = 2.0
PROCESSING_INTERVAL = 1
SILENCE_THRESHOLD = 25
MIN_SPEECH_DURATION = 0.3

# === Prediction class ===
class PredictionWithConfidence:
    def __init__(self, speaker, confidence, timestamp):
        self.speaker = speaker
        self.confidence = confidence
        self.timestamp = timestamp

# === Helper functions ===
def has_speech(audio_np: np.ndarray, threshold=SILENCE_THRESHOLD):
    """Check if audio segment contains speech based on RMS energy"""
    rms = np.sqrt(np.mean(audio_np**2))
    return rms > threshold

def get_audio_level(audio_np: np.ndarray):
    """Get normalized audio level (0-100)"""
    if len(audio_np) == 0:
        return 0
    rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
    normalized = min(100, (rms / 3000) * 100)
    return int(normalized)

def identify_speaker_with_confidence(audio_np: np.ndarray, sample_rate: int):
    """Predict speaker and return confidence score"""
    try:
        if not has_speech(audio_np):
            return None, 0.0

        start_time = time.time()

        # Write to temp WAV
        wav_path = Path("temp_clip.wav")
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())

        # Clean + extract features
        cleaned_path = Path("temp_clean.wav")
        step1.clean_audio(wav_path, cleaned_path)
        features = step4.extract_features(cleaned_path)

        if features is None:
            return None, 0.0

        X = np.array([v for k, v in features.items() if "mfcc" in k]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        pred = model.predict(X_scaled)[0]
        
        try:
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(np.max(proba))
        except:
            confidence = 0.7

        proc_time = (time.time() - start_time) * 1000
        state.processing_times.append(proc_time)

        return pred, confidence

    except Exception as e:
        print(f"[ERROR] Speaker identification failed: {e}")
        return None, 0.0

def get_smoothed_prediction():
    """Get weighted prediction from recent predictions based on confidence"""
    if not state.recent_predictions:
        return "Detecting...", 0.0
    
    weighted_votes = {}
    total_weight = 0
    
    for i, pred_obj in enumerate(state.recent_predictions):
        recency_weight = (i + 1) / len(state.recent_predictions)
        weight = pred_obj.confidence * recency_weight
        
        if pred_obj.speaker not in weighted_votes:
            weighted_votes[pred_obj.speaker] = 0
        weighted_votes[pred_obj.speaker] += weight
        total_weight += weight
    
    if not weighted_votes:
        return "Detecting...", 0.0
        
    best_speaker = max(weighted_votes, key=weighted_votes.get)
    confidence = weighted_votes[best_speaker] / total_weight if total_weight > 0 else 0.0
    
    return best_speaker, confidence

def prediction_worker():
    """Background worker to process speaker predictions without blocking UI"""
    while True:
        try:
            audio_data, sample_rate, timestamp = state.prediction_queue.get(timeout=1)
            
            if audio_data is None:
                break
                
            prediction, confidence = identify_speaker_with_confidence(audio_data, sample_rate)
            
            if prediction:
                pred_obj = PredictionWithConfidence(prediction, confidence, timestamp)
                state.recent_predictions.append(pred_obj)
                state.current_speaker, state.current_confidence = get_smoothed_prediction()
                
                avg_proc_time = np.mean(state.processing_times) if state.processing_times else 0
                print(f"[PREDICTION @ {timestamp:.1f}s] "
                      f"Raw: {prediction} ({confidence:.2%}) | "
                      f"Smoothed: {state.current_speaker} ({state.current_confidence:.2%}) | "
                      f"Proc: {avg_proc_time:.0f}ms")
        
        except Exception as e:
            if "Empty queue" not in str(e):
                print(f"[ERROR] Prediction worker error: {e}")

# Start background prediction thread
prediction_thread = threading.Thread(target=prediction_worker, daemon=True)
prediction_thread.start()

def process_audio(audio):
    """Main audio processing function"""
    if audio is None:
        return "Listening...", f'Speaking: {state.current_speaker} \n ({state.current_confidence:.1%} confidence)'

    try:
        sample_rate, audio_data = audio

        # Initialize buffer on first run
        if state.buffer is None:
            state.sample_rate = sample_rate
            clip_samples = int(CLIP_LENGTH * sample_rate)
            state.buffer = deque(maxlen=clip_samples)
            print(f"[INFO] Initialized audio buffer: {sample_rate}Hz, {clip_samples} samples ({CLIP_LENGTH}s)")
        elif state.sample_rate != sample_rate:
            state.sample_rate = sample_rate
            clip_samples = int(CLIP_LENGTH * sample_rate)
            state.buffer = deque(maxlen=clip_samples)
            print(f"[INFO] Sample rate changed! Re-initialized buffer: {sample_rate}Hz")

        # Normalize dtype
        if audio_data.dtype in (np.float32, np.float64):
            audio_data = (audio_data * 32767).astype(np.int16)

        # Mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(np.int16)

        # Get audio level for monitoring
        audio_level = get_audio_level(audio_data)
        if audio_level > 5:
            print(f"[AUDIO] Level: {audio_level}%")

        # Add to rolling buffer
        state.buffer.extend(audio_data)

        now = time.time()

        # Calculate samples needed
        clip_samples = int(CLIP_LENGTH * state.sample_rate)
        interval_samples = int(PROCESSING_INTERVAL * state.sample_rate)

        # Process every 0.5 seconds if we have enough data
        if now - state.last_process_time >= PROCESSING_INTERVAL and len(state.buffer) >= clip_samples:
            clip_audio = np.array(list(state.buffer)[-clip_samples:])
            
            print(f"[DEBUG] Clip: {len(clip_audio)} samples at {state.sample_rate}Hz = {len(clip_audio)/state.sample_rate:.2f}s")
            
            if state.prediction_queue.qsize() < 3:
                state.prediction_queue.put((clip_audio.copy(), state.sample_rate, now))
            else:
                print("[WARN] Prediction queue full, skipping frame")
            
            state.last_process_time = now

        # Speech recognition
        if len(state.buffer) >= interval_samples:
            transcribe_audio = np.array(list(state.buffer))
            
            if has_speech(transcribe_audio):
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(state.sample_rate)
                    wf.writeframes(transcribe_audio.tobytes())
                wav_io.seek(0)

                try:
                    with sr.AudioFile(wav_io) as source:
                        audio_chunk = recognizer.record(source)
                        text = recognizer.recognize_google(audio_chunk, language="en-US").strip()

                    if text:
                        if now - state.last_update_time > state.pause_threshold:
                            state.recent_transcript.clear()
                        state.recent_transcript.append(text)
                        state.last_update_time = now
                        print(f"[TRANSCRIPT] {text}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"[ERROR] Speech recognition service error: {e}")
                except Exception as e:
                    print(f"[ERROR] Transcription error: {e}")

        # Return current state
        transcript_text = " ".join(state.recent_transcript) if state.recent_transcript else "Listening..."
        speaker_text = f"Speaking: {state.current_speaker}\n({state.current_confidence:.1%} confidence)"
        
        return transcript_text, speaker_text

    except Exception as e:
        print(f"[ERROR] Audio processing error: {e}")
        import traceback
        traceback.print_exc()
        transcript_text = " ".join(state.recent_transcript) if state.recent_transcript else "Listening..."
        speaker_text = f"{state.current_speaker}\nConfidence: {state.current_confidence:.1%}"
        return transcript_text, speaker_text

def cleanup():
    """Cleanup resources on shutdown"""
    print("[INFO] Shutting down...")
    state.prediction_queue.put((None, None, None))
    prediction_thread.join(timeout=2)
    
    for temp_file in ["temp_clip.wav", "temp_clean.wav"]:
        try:
            Path(temp_file).unlink(missing_ok=True)
        except:
            pass

import atexit
atexit.register(cleanup)


# === Custom CSS ===
css = """
.gradio-container {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 18px !important;
    height: 100vh !important;
    width: 100vw !important;
    max-width: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}
.block, .form, .input-container, .show_textbox_border, textarea {
    background: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 18px !important;
    color: #000000 !important;
}
.speaker-box {
    width: 100vw !important;
    max-width: 100% !important;
    margin: 20px auto !important;
    text-align: left !important;
}
button {
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
footer, .footer, .button-wrap.svelte-10cpz3p {
    display: none !important;
}
"""

# === Gradio UI ===
with gr.Blocks(
    theme=gr.themes.Default(font="serif"),
    css=css,
    title="Live Speech + Speaker ID Demo"
) as demo:
    
    webcam = gr.Image(
        sources=["webcam"],
        streaming=True,
        show_label=False,
        mirror_webcam=True,
        height=400,
        width=600,
    )

    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        label=None,
        show_label=False,
        show_download_button=False,
    )

    # Speaker recognition at the top (centered)
    speaker_output = gr.Textbox(
        label="Speaking:",
        placeholder="Predicted speaker will appear here...",
        lines=3,
        max_lines=3,
        show_label=False,
        elem_classes=["speaker-box"]
    )

    # Commented out: Speech recognition (uncomment and add HuggingFace credentials in .env to use)
    # transcription_output = gr.Textbox(
    #     label="",
    #     placeholder="Live captions will appear here...",
    #     lines=6,
    #     max_lines=15,
    #     show_label=False
    # )

    # Stream outputs (only speaker prediction now)
    audio_input.stream(
        fn=lambda audio: process_audio(audio)[1] if audio else f"Speaking: {state.current_speaker}\n({state.current_confidence:.1%} confidence)",
        inputs=audio_input,
        outputs=[speaker_output],
        stream_every=0.5
    )

if __name__ == "__main__":
    print("[INFO] Starting Gradio demo...")
    print(f"[INFO] Model loaded: {cfg['MODEL_DIR'] / 'speaker_recognition_knn.pkl'}")
    print(f"[INFO] Processing: {CLIP_LENGTH}s clips every {PROCESSING_INTERVAL}s")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )