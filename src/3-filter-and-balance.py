# src/3-filter-and-balance-data.py
import shutil
import random
import librosa
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import track
from config import load_config

# setup console
console = Console()

cfg = load_config()

random.seed(cfg["RANDOM_SEED"])

def analyze_clip_quality(file_path):
    """
    Analyze a clip's audio quality using RMS energy.
    
    Args:
        file_path: pathlib.Path, path to the audio clip
    
    Returns:
        dict with quality metrics or None if error
    """
    try:
        y, sr = librosa.load(file_path, sr=cfg["SAMPLE_RATE"])
        
        # calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        
        return {
            'rms_mean': rms_mean,
            'passes_threshold': rms_mean >= cfg["RMS_THRESHOLD"]
        }
        
    except Exception as e:
        console.print(f"[red]error analyzing {file_path.name}: {e}[/red]")
        return None


def filter_clips_by_quality(clips_dir):
    """
    Filter clips based on RMS energy threshold.
    
    Args:
        clips_dir: pathlib.Path, directory containing speaker clip folders
    
    Returns:
        dict mapping speaker_id to list of (clip_path, quality_metrics) tuples
    """
    filtered_clips = {}
    total_clips = 0
    filtered_out = 0
    
    # process each speaker directory
    for speaker_dir in clips_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        console.print(f"[cyan]filtering clips for {speaker_id}[/cyan]")
        
        # find all wav files for this speaker
        wav_files = list(speaker_dir.glob("*.wav"))
        speaker_clips = []
        
        # analyze each clip
        for wav_file in track(wav_files, description=f"analyzing {speaker_id} clips"):
            total_clips += 1
            quality = analyze_clip_quality(wav_file)
            
            if quality and quality['passes_threshold']:
                speaker_clips.append((wav_file, quality))
            else:
                filtered_out += 1
        
        filtered_clips[speaker_id] = speaker_clips
        console.print(f"[green]kept {len(speaker_clips)}/{len(wav_files)} clips for {speaker_id}[/green]")
    
    console.print(f"[cyan]total clips analyzed: {total_clips}[/cyan]")
    console.print(f"[cyan]clips filtered out: {filtered_out}[/cyan]")
    console.print(f"[cyan]clips kept: {total_clips - filtered_out}[/cyan]")
    
    return filtered_clips


def balance_dataset(filtered_clips):
    """
    Balance the dataset by ensuring equal number of clips per speaker.
    
    Args:
        filtered_clips: dict mapping speaker_id to list of clip tuples
    
    Returns:
        dict with balanced clips per speaker
    """
    if not filtered_clips:
        return {}
    
    # find minimum number of clips across speakers
    clip_counts = {speaker: len(clips) for speaker, clips in filtered_clips.items()}
    min_clips = min(clip_counts.values())
    
    console.print(f"[cyan]clip counts per speaker: {dict(clip_counts)}[/cyan]")
    console.print(f"[cyan]balancing to {min_clips} clips per speaker[/cyan]")
    
    if min_clips == 0:
        console.print(f"[red]error: at least one speaker has no clips after filtering[/red]")
        return {}
    
    balanced_clips = {}
    
    # randomly sample clips for each speaker
    for speaker_id, clips in filtered_clips.items():
        if len(clips) >= min_clips:
            # randomly sample clips
            balanced_clips[speaker_id] = random.sample(clips, min_clips)
            console.print(f"[green]sampled {min_clips} clips for {speaker_id}[/green]")
        else:
            # this shouldn't happen given our min_clips calculation, but just in case
            balanced_clips[speaker_id] = clips
            console.print(f"[yellow]warning: {speaker_id} has fewer clips than target[/yellow]")
    
    return balanced_clips


def save_balanced_clips(balanced_clips, output_dir):
    """
    Save balanced clips to the output directory and create manifest csv.
    
    Args:
        balanced_clips: dict with balanced clips per speaker
        output_dir: pathlib.Path, output directory for balanced clips
    
    Returns:
        list of dictionaries with clip metadata for manifest
    """
    manifest_data = []
    total_copied = 0
    
    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for speaker_id, clips in balanced_clips.items():
        console.print(f"[cyan]saving {len(clips)} clips for {speaker_id}[/cyan]")
        
        # create speaker output directory
        speaker_output_dir = output_dir / speaker_id
        speaker_output_dir.mkdir(exist_ok=True)
        
        # save each clip
        for clip_path, quality in track(clips, description=f"saving {speaker_id} clips"):
            # generate new filename to avoid conflicts
            output_path = speaker_output_dir / clip_path.name
            
            # copy file
            shutil.copy2(clip_path, output_path)
            total_copied += 1
            
            # add to manifest
            manifest_data.append({
                "clip_path": str(output_path),
                "original_path": str(clip_path),
                "speaker_id": speaker_id,
                "script_id": clip_path.stem.split('_')[0],  # extract script_id from filename
                "rms_mean": quality['rms_mean'],
                "duration": 2.0,  # clips are 2 seconds from splitting step
                "balanced": True
            })
    
    console.print(f"[green]copied {total_copied} total clips[/green]")
    return manifest_data


def main():
    """
    Main function to filter and balance clip dataset.
    
    Args:
        none
    
    Returns:
        saves balanced clips and manifest file

    Pipeline Step:
        3/5
    
    Expects:
        processed clips in PROCESSED_CLIPS_DIR
    """
    
    if not cfg["PROCESSED_CLIPS_DIR"].exists():
        console.print(f'[red]error: processed clips directory not found: {cfg["PROCESSED_CLIPS_DIR"]}[/red]')
        return
    
    # find speaker directories
    speaker_dirs = [d for d in cfg["PROCESSED_CLIPS_DIR"].iterdir() if d.is_dir()]
    
    if not speaker_dirs:
        console.print(f'[red]error: no speaker directories found in {cfg["PROCESSED_CLIPS_DIR"]}[/red]')
        return
    
    console.rule("[bold green]starting clip filtering and balancing[/bold green]")
    console.print(f"[cyan]found {len(speaker_dirs)} speaker(s) to process[/cyan]")
    console.print(f'[cyan]RMS threshold: {cfg["RMS_THRESHOLD"]}[/cyan]')
    
    # step 1: filter clips by quality
    console.rule("[bold cyan]step 1: filtering clips by quality[/bold cyan]")
    filtered_clips = filter_clips_by_quality(cfg["PROCESSED_CLIPS_DIR"])
    
    if not filtered_clips:
        console.print(f"[red]error: no clips passed quality filtering[/red]")
        return
    
    # step 2: balance dataset
    console.rule("[bold cyan]step 2: balancing dataset[/bold cyan]")
    balanced_clips = balance_dataset(filtered_clips)
    
    if not balanced_clips:
        console.print(f"[red]error: failed to balance dataset[/red]")
        return
    
    # step 3: copy balanced clips and create manifest
    console.rule("[bold cyan]step 3: copying balanced clips[/bold cyan]")
    
    # clean output directory if it exists
    if cfg["BALANCED_CLIPS_DIR"].exists():
        console.print(f'[yellow]cleaning existing output directory: {cfg["BALANCED_CLIPS_DIR"]}[/yellow]')
        shutil.rmtree(cfg["BALANCED_CLIPS_DIR"])
    
    manifest_data = save_balanced_clips(balanced_clips, cfg["BALANCED_CLIPS_DIR"])
    
    # save balanced manifest
    if manifest_data:
        console.print(f"[cyan]saving balanced manifest with {len(manifest_data)} clips...[/cyan]")
        df = pd.DataFrame(manifest_data)
        cfg["BALANCED_MANIFEST_FILE"].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg["BALANCED_MANIFEST_FILE"], index=False)
        console.print(f'[bold green]balanced manifest saved: {cfg["BALANCED_MANIFEST_FILE"]}[/bold green]')
    
    # final summary
    console.rule("[bold green]filtering and balancing complete![/bold green]")
    
    total_speakers = len(balanced_clips)
    clips_per_speaker = len(next(iter(balanced_clips.values()))) if balanced_clips else 0
    total_balanced_clips = total_speakers * clips_per_speaker
    
    console.print(f"[bold green]speakers: {total_speakers}[/bold green]")
    console.print(f"[bold green]clips per speaker: {clips_per_speaker}[/bold green]")
    console.print(f"[bold green]total balanced clips: {total_balanced_clips}[/bold green]")
    console.print(f'[bold green]balanced clips saved to: {cfg["BALANCED_CLIPS_DIR"]}[/bold green]')
    console.print(f'[bold green]RMS threshold used: {cfg["RMS_THRESHOLD"]}[/bold green]')


if __name__ == "__main__":
    main()