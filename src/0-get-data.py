import json
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress
import time
from config import PROMPTS_FILE, RAW_RECORDINGS_DIR, MANIFEST_FILE

RECORDING_DURATION = 60
COUNTDOWN_DURATION = 10

console = Console()


def record_with_progress(duration, filename):
    """Record audio at the device's native sample rate."""
    device_info = sd.query_devices(kind='input')
    sample_rate = int(device_info['default_samplerate'])

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("[cyan]recording...", total=duration)
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        start = time.time()
        while (elapsed := time.time() - start) < duration:
            progress.update(task, completed=elapsed)
            time.sleep(0.1)
        sd.wait()

    sf.write(filename, audio, sample_rate)


def main():
    """Main script for recording all prompts."""
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        all_prompts = json.load(f)

    # ask for speaker name
    speaker_id = Prompt.ask("enter your name (e.g. john)").strip().lower().replace(" ", "_")

    # ask for spoken language (used to select the correct E prompt)
    console.rule("[bold blue]language selection[/bold blue]")
    console.print("The last prompt will be in a language of your choice. Please make a selection:")
    console.print("[1] Kazakh (E1)\n[2] French (E2)\n[3] Ukrainian (E3)\n[4] Russian (E4)")
    lang_choice = Prompt.ask("enter the number corresponding to your language", choices=["1", "2", "3", "4"])
    selected_e = f"E{lang_choice}"

    # filter prompts to include A, B, C, D and selected E
    prompts = [p for p in all_prompts if p["id"] in ["A", "B", "C", "D", selected_e]]

    # create speaker folder
    session_dir = RAW_RECORDINGS_DIR / speaker_id
    session_dir.mkdir(parents=True, exist_ok=True)

    for script in prompts:
        console.rule(f"[bold green]{script['title']}[/bold green]")
        console.print(f"[yellow]instruction:[/yellow] {script['instruction']}")
        console.rule()

        if isinstance(script["content"], list):
            for line in script["content"]:
                console.print(f"- {line}")
        else:
            console.print(script["content"])

        console.rule()

        out_file = session_dir / f"{script['id']}.wav"
        console.print(f"[cyan]recording will start in {COUNTDOWN_DURATION} seconds... get ready[/cyan]")
        time.sleep(COUNTDOWN_DURATION)

        record_with_progress(RECORDING_DURATION, out_file)
        console.print(f"[bold green]recording saved: {out_file}[/bold green]")

    console.rule("[bold green]all done! thank you for your recordings![/bold green]")
    console.print(f"[bold green]{speaker_id}'s raw recordings ready at {session_dir}.[/bold green] saved manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
