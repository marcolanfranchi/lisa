# src/2-split-clips.py
import uuid
import librosa
import soundfile as sf
import pandas as pd
import shutil
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from config import load_config

# setup console
console = Console()

cfg = load_config()

def split_audio(file_path, speaker_id, script_id):
    """
    Split cleaned audio into overlapping clips and save them.
    
    Args:
        file_path: pathlib.Path, path to the cleaned audio file
        speaker_id: str, identifier for the speaker
        script_id: str, identifier for the script/prompt
    
    Returns:
        a list of dictionaries with clip metadata
    """
    try:
        # load audio
        y, sr = librosa.load(file_path, sr=cfg["SAMPLE_RATE"])
        clip_len = int(cfg["CLIP_LEN"] * sr)
        step = int(cfg["STEP"] * sr)

        clips = []
        
        # create clips with sliding window
        for start in range(0, len(y) - clip_len, step):
            end = start + clip_len
            clip = y[start:end]

            # generate unique clip ID
            clip_id = str(uuid.uuid4())[:8]
            out_name = f"{script_id}_{clip_id}.wav"
            out_path = cfg["PROCESSED_CLIPS_DIR"] / speaker_id / out_name

            # ensure output directory exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # save clip
            sf.write(out_path, clip, sr)
            
            # add metadata
            clips.append({
                "clip_path": str(out_path),
                "speaker_id": speaker_id,
                "script_id": script_id,
                "source_file": str(file_path),
                "start_time": start / sr,
                "end_time": end / sr,
                "duration": len(clip) / sr
            })

        return clips
        
    except Exception as e:
        console.print(f"[red]error splitting {file_path.name}: {e}[/red]")
        return []


def main():
    """
    main function to split all cleaned recordings into clips.
    
    Args:
        none
    
    Returns:
        saves processed clips and a manifest file

    Pipeline Step:
        2/5
    
    Expects:
        cleaned recordings in CLEANED_RECORDINGS_DIR
    """

    if not cfg["CLEANED_RECORDINGS_DIR"].exists():
        console.print(f'[red]error: cleaned recordings directory not found: \
                      {cfg["CLEANED_RECORDINGS_DIR"]}[/red]')
        return
    
    # find all speaker directories
    speaker_dirs = [d for d in cfg["CLEANED_RECORDINGS_DIR"].iterdir() if d.is_dir()]
    
    if not speaker_dirs:
        console.print(f'[red]error: no speaker directories found in \
                      {cfg["CLEANED_RECORDINGS_DIR"]}[/red]')
        return
    
    console.print(f"[cyan]found {len(speaker_dirs)} speaker(s) with cleaned recordings to process: \
                  {', '.join([d.name for d in speaker_dirs])}[/cyan]")
    
    all_clips = []
    total_files = 0
    total_clips = 0
    
    # process each speaker
    for speaker_dir in speaker_dirs:

        speaker_id = speaker_dir.name
        console.rule(f"[bold green]Splitting clips for speaker: {speaker_id}[/bold green]")
        
        # remove existing processed clips for this speaker to avoid duplicates
        processed_speaker_dir = cfg["PROCESSED_CLIPS_DIR"] / speaker_dir.name
        if processed_speaker_dir.exists():
            console.print(f"[yellow]clearing existing split clips directory: {processed_speaker_dir}[/yellow]")
            shutil.rmtree(processed_speaker_dir)

        # find all wav files for this speaker
        wav_files = list(speaker_dir.glob("*.wav"))
        
        if not wav_files:
            console.print(f"[yellow]no .wav files found for {speaker_id}[/yellow]")
            continue
        
        console.print(f"[cyan]found {len(wav_files)} cleaned recordings for {speaker_id}[/cyan]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(f"splitting audio files for {speaker_id}", total=len(wav_files))
            # split each file with progress tracking
            speaker_clips = 0
            for wav_file in wav_files:
                total_files += 1
                
                # extract script_id from filename (remove .wav extension)
                script_id = wav_file.stem
            
                # split audio into clips
                clips = split_audio(wav_file, speaker_id, script_id)
                progress.advance(task)
                all_clips.extend(clips)
                total_clips += len(clips)
                speaker_clips += len(clips)
            
                if not clips:
                    console.print(f"[yellow]  {wav_file.name}: 0 clips [/yellow]")

            # print total clips for this speaker
            console.print(f"[cyan]  {speaker_id}: generated {speaker_clips} clips from {len(wav_files)} files[/cyan")

    console.rule(f"[bold green]Done splitting clips[/bold green]")
    
    # save manifest file
    if all_clips:
        console.print(f"[cyan]saving manifest with {len(all_clips)} clips...[/cyan]")
        df = pd.DataFrame(all_clips)
        cfg["MANIFEST_FILE"].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg["MANIFEST_FILE"], index=False)
        console.print(f'[bold green]manifest saved: {cfg["MANIFEST_FILE"]}[/bold green]')
    else:
        console.print(f"[red]no clips generated, manifest not created[/red]")
    
    # summary
    console.print(f"[bold green]processed {total_files} files into {total_clips} clips[/bold green]")
    console.print(f'[bold green]processed clips saved to: {cfg["PROCESSED_CLIPS_DIR"]}[/bold green]')


if __name__ == "__main__":
    main()
