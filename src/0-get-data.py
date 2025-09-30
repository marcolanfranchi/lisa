# 0-get-data.py
import json
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress
import time
from config import PROMPTS_FILE, RAW_RECORDINGS_DIR, MANIFEST_FILE

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
RECORDING_DURATION = 60  # Time given to read each prompt (in seconds)
COUNTDOWN_DURATION = 10  # Time given to get ready before recording starts (in seconds)
# ----------------------------------------

# setup console
console = Console()


def record_with_progress(duration, filename):
    """ 
    Record audio with progress bar feedback.

    Args:
        duration: int, recording duration in seconds
        filename: str, path to save the recording
    
    Returns:
        saves the recording to the specified filename
    """
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("[cyan]recording...", total=duration)
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        for _ in range(duration):
            time.sleep(1)
            progress.update(task, advance=1)
        sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)


def main():
    """
    Main script to handle the recording process.

    Args:
        none

    Returns:
        saves raw recordings and updates manifest file
    
    Pipeline Step:
        0/5

    Expects:
        nothing, starts fresh recording session and saves data for next steps
    """

    # load read-out-loud prompts
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    # get speaker ID (first name)
    speaker_id = Prompt.ask("enter your name (e.g. john)").strip().lower().replace(" ", "_")

    # create directory for speaker
    session_dir = RAW_RECORDINGS_DIR / speaker_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # iterate over 'read-out-loud' prompts and record
    for script in prompts:
        console.rule(f"[bold green]{script['title']}[/bold green]")
        console.print(f"[yellow]instruction:[/yellow] {script['instruction']}")
        console.rule()

        # display script content (handle list or str)
        if isinstance(script["content"], list):
            for line in script["content"]:
                console.print(f"- {line}")
        else:
            console.print(script["content"])

        console.rule()

        # prepare file path for raw recording output
        out_file = session_dir / f"{script['id']}.wav"

        console.print(f"[cyan]recording will start in {COUNTDOWN_DURATION} seconds... get ready[/cyan]")
        time.sleep(COUNTDOWN_DURATION)

        # record audio with progress bar
        record_with_progress(RECORDING_DURATION, out_file)
        console.print(f"[bold green]recording saved: {out_file}[/bold green]")

    # once for-loop is done, show thank you message
    console.rule("[bold green]all done! thank you for your recordings![/bold green]")
    console.print(f"[bold green]{speaker_id}'s raw recordings ready at {session_dir}.[/bold green] saved manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
