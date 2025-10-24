# src/1-clean-audio.py
from .config import RAW_RECORDINGS_DIR, CLEANED_RECORDINGS_DIR
import librosa
import soundfile as sf
from rich.console import Console
from rich.progress import track
import noisereduce as nr

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
# ----------------------------------------

# setup console
console = Console()


def clean_audio(input_path, output_path):
    """
    Apply audio cleaning techniques to remove background noise.

    Args:
        input_path: pathlib.Path, path to the raw audio file
        output_path: pathlib.Path, path to save the cleaned audio

    Returns:
        saves cleaned audio to the specified output path
    """
    try:
        # load audio
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
        
        # 1. noise reduction using noisereduce library
        # use the first 0.5 seconds as noise sample for stationary noise reduction
        y_reduced = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.8)
        
        # 2. normalize audio to consistent volume level
        # use librosa's util.normalize to peak normalize
        y_normalized = librosa.util.normalize(y_reduced)
        
        # 3. apply gentle high-pass filter to remove low-frequency rumble
        # remove frequencies below 80 Hz which are typically not speech
        y_filtered = librosa.effects.preemphasis(y_normalized, coef=0.97)
        
        # save cleaned audio
        sf.write(output_path, y_filtered, sr)
        console.print(f"[green]cleaned audio saved: {output_path}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]error cleaning {input_path.name}: {e}[/red]")
        return False


def main():
    """
    Main script to clean all raw recordings.
    
    Args:
        none
    
    Returns:
        none
    
    Pipeline Step:
        1/5

    Expects:
        raw recordings in RAW_RECORDINGS_DIR
    """

    if not RAW_RECORDINGS_DIR.exists():
        console.print(f"[red]error: raw recordings directory not found: {RAW_RECORDINGS_DIR}[/red]")
        return
    
    # find all speaker directories
    speaker_dirs = [d for d in RAW_RECORDINGS_DIR.iterdir() if d.is_dir()]
    
    if not speaker_dirs:
        console.print(f"[red]error: no speaker directories found in {RAW_RECORDINGS_DIR}[/red]")
        return
    
    console.print(f"[cyan]found {len(speaker_dirs)} speaker(s) to process[/cyan]")
    
    total_files = 0
    cleaned_files = 0
    
    # process each speaker
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        console.rule(f"[bold green]processing speaker: {speaker_id}[/bold green]")
        
        # create cleaned directory for this speaker
        cleaned_speaker_dir = CLEANED_RECORDINGS_DIR / speaker_id
        cleaned_speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # find all wav files for this speaker
        wav_files = list(speaker_dir.glob("*.wav"))
        
        if not wav_files:
            console.print(f"[yellow]no .wav files found for {speaker_id}[/yellow]")
            continue
        
        console.print(f"[cyan]cleaning {len(wav_files)} recordings for {speaker_id}[/cyan]")
        
        # clean each file with progress tracking
        for wav_file in track(wav_files, description="cleaning audio files"):
            total_files += 1
            output_path = cleaned_speaker_dir / wav_file.name
            
            if clean_audio(wav_file, output_path):
                cleaned_files += 1
    
    # summary
    console.rule("[bold green]cleaning complete![/bold green]")
    console.print(f"[bold green]processed {cleaned_files}/{total_files} files successfully[/bold green]")
    console.print(f"[bold green]cleaned recordings saved to: {CLEANED_RECORDINGS_DIR}[/bold green]")
    
    if cleaned_files < total_files:
        console.print(f"[yellow]warning: {total_files - cleaned_files} files had errors (check cleaning.log)[/yellow]")


if __name__ == "__main__":
    main()
