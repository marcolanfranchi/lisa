# 0-get-data.py

import os
import json
import dotenv
import yt_dlp
from pathlib import Path
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress
from src.config import load_config
#from config import DIARIZED_RECORDINGS_DIR, NEW_MANIFEST_FILE 

# ---------------- CONFIG ----------------

# ----------------------------------------

# setup console
console = Console()
dotenv.load_dotenv()

#load config.yaml
cfg = load_config()


# load HuggingFace token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    console.print("[bold red]Error:[/bold red] HUGGINGFACE_TOKEN not found in environment.")
    exit(1)

# Initialize diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)


def download_youtube_audio(url, output_dir):
    """
    Download a YouTube video's audio track as mp3.

    Args:
        url (str): YouTube video URL
        output_dir (Path): destination directory

    Returns:
        Path to downloaded mp3 file
    """

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': cfg["AUDIO_FORMAT"],
            'preferredquality': '192',
        }],
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).rsplit('.', 1)[0] + f".{cfg["AUDIO_FORMAT"]}"
        return Path(filename)

def diarize_audio(audio_path, output_dir):
    """
    Run speaker diarization and save each speaker segment as individual WAVs.

    Args: 
        audio_path (Path): path to mp3/wav audio file
        output_dir (Path): directory to store speaker clips

    Returns:
        list of dicts with diarization segment metadata
    """

    console.rule(f"[bold green]Running speaker diarization on {audio_path.name}[/bold green]")

    # Convert MP3 to temporary WAV first - prevents sample mismatch errors
    wav_path = output_dir / (audio_path.stem + "_clean.wav")
    AudioSegment.from_file(audio_path).set_frame_rate(16000).export(wav_path, format="wav")
    audio_path = wav_path

    # Run diarization pipeline
    with ProgressHook() as hook:
        output = pipeline(audio_path, hook=hook)

    # Access speaker_diarization attribute directly
    diarization = output.speaker_diarization

    # Load audio for splitting
    audio = AudioSegment.from_file(audio_path)
    segment_data = []

    # Iterate over each (turn, speaker) pair
    for i, (turn, speaker) in enumerate(diarization, start=1):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        speaker_dir = output_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        clip = audio[start_ms:end_ms]
        out_file = speaker_dir / f"segment_{i:03d}.wav"
        clip.export(out_file, format="wav")
        segment_data.append({
            "segment_id": i,
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "path": str(out_file)
        })

    return segment_data


def main():
    """
    Main script to collect speaker segments from a YouTube video.

    Places diarized speaker clips in DIARIZED_RECORDINGS_DIR/<video_id>/
    Updates NEW_MANIFEST_FILE with metadata about the video and segments.
    """

    console.rule("[bold blue]YouTube Audio Speaker Diarization[/bold blue]")
    url = Prompt.ask("Enter YouTube video URL").strip()

    # create base dir for this session
    video_id = url.split("v=")[-1].split("&")[0]
    session_dir = cfg["DIARIZED_RECORDINGS_DIR"] / video_id.replace("/", "_") # sanitize slashes to avoid nested dirs
    session_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Download audio
    console.print("[yellow]Downloading audio...[/yellow]")
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("[cyan]downloading...", total=None)
        audio_path = download_youtube_audio(url, session_dir)
        progress.update(task, completed=1)

    console.print(f"[bold green]Downloaded:[/bold green] {audio_path}")

    # STEP 2: Diarization
    segments = diarize_audio(audio_path, session_dir)

    # STEP 3: Save manifest
    manifest_entry = {
        "video_id": video_id,
        "source_url": url,
        "audio_file": str(audio_path),
        "num_segments": len(segments),
        "segments": segments
    }

    if not cfg["NEW_MANIFEST_FILE"].exists():
        manifest = []
    else:
        with open(cfg["NEW_MANIFEST_FILE"]) as f:
            manifest = json.load(f)

    manifest.append(manifest_entry)

    with open(cfg["EW_MANIFEST_FILE"], "w") as f:
        json.dump(manifest, f, indent=2)

    console.rule("[bold green]All done![/bold green]")
    console.print(f"[bold cyan]Speaker clips ready in:[/bold cyan] {session_dir}")
    console.print(f"[bold cyan]Manifest updated:[/bold cyan] {cfg["NEW_MANIFEST_FILE"]}")


if __name__ == "__main__":
    main()
