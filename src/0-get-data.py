# 0-get-data.py
import json
import uuid
import logging
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import librosa
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress
import time

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
CLIP_LEN = 1.5   # seconds
STEP = 0.75      # seconds (50% overlap)
RAW_DIR = Path("data/generated/raw_recordings")
PROC_DIR = Path("data/generated/processed_clips")
PROMPTS_FILE = Path("data/generated/recording-prompts.json")
MANIFEST_FILE = Path("data/generated/manifest.csv")
# ----------------------------------------

# setup console and logging
console = Console()
logging.basicConfig(
    filename="recording.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def countdown(seconds, message):
    """show a countdown with progress bar"""
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task(f"[cyan]{message}", total=seconds)
        for _ in range(seconds):
            time.sleep(1)
            progress.update(task, advance=1)


def record_with_progress(duration, filename):
    """record audio with progress bar feedback"""
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("[cyan]recording...", total=duration)
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        for _ in range(duration):
            time.sleep(1)
            progress.update(task, advance=1)
        sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    logging.info(f"saved recording: {filename}")


def split_audio(file_path, speaker_id, script_id):
    """split raw audio into overlapping clips and save them"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    clip_len = int(CLIP_LEN * sr)
    step = int(STEP * sr)

    clips = []
    for start in range(0, len(y) - clip_len, step):
        end = start + clip_len
        clip = y[start:end]

        # trim silence
        clip, _ = librosa.effects.trim(clip, top_db=25)
        if len(clip) < sr * 0.4:  # discard if too short
            continue

        clip_id = str(uuid.uuid4())[:8]
        out_name = f"{script_id}_{clip_id}.wav"
        out_path = PROC_DIR / speaker_id / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(out_path, clip, sr)
        clips.append({
            "clip_path": str(out_path),
            "speaker_id": speaker_id,
            "script_id": script_id,
            "source_file": str(file_path),
            "start_time": start / sr,
            "end_time": end / sr
        })

    return clips


def main():
    # load prompts
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    speaker_id = Prompt.ask("enter your name (e.g. marco)")

    session_dir = RAW_DIR / speaker_id
    session_dir.mkdir(parents=True, exist_ok=True)

    all_clips = []

    for script in prompts:
        console.rule(f"[bold green]{script['title']}[/bold green]")
        # console.print(f"[yellow]reason:[/yellow] {script['reason']}")
        console.print(f"[yellow]instruction:[/yellow] {script['instruction']}")
        console.rule()

        if isinstance(script["content"], list):
            for line in script["content"]:
                console.print(f"- {line}")
        else:
            console.print(script["content"])

        console.rule()
        # audio recording duration to 60 seconds
        duration = 60
        out_file = session_dir / f"{script['id']}.wav"

        console.print(f"[cyan]recording will start in 5 seconds... get ready[/cyan]")
        time.sleep(5)

        record_with_progress(duration, out_file)

        # split into clips
        console.print(f"[magenta]splitting {out_file} into clips...[/magenta]")
        clips = split_audio(out_file, speaker_id, script["id"])
        all_clips.extend(clips)

    # save manifest
    df = pd.DataFrame(all_clips)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_FILE, index=False)
    console.print(f"[bold green]dataset ready.[/bold green] saved manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
