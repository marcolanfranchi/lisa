# L.I.S.A. (Labeled Identification of Speech Audio)

An end-to-end machine learning pipeline that classifies a speaker by their voice.

## Overview

### System Diagram

![System Diagram](images/diagramTransparentDarkBG.png)

### Project Structure

```
speaker-recognition/
│
├── data/                       # Data
│ └── generated/                # Generated dataset of voice recordings
│   └── raw_recordings/         # Folders for each speaker with raw recordings
│   └── cleaned_recordings/     # Folders for each speaker with cleaned recordings
│   └── processed_clips/        # Folders for each speaker with split audio clips
│   └── manifest.csv            # Table describing the dataset files
│ └── recording-prompts.json    # Voice recording instructions
│
├── src/                        # Source code
│ └── 0-get-data.py             # Starts the data generation process
│ └── 1-clean-audio.py          # 
│ └── 2-split-clips.py          # 
│ └── 3-extract-features.py     # 
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/marcolanfranchi/speaker-recognition.git
cd speaker-recognition
```

### 2. Create and activate environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate your dataset 
This step needs to be done for 3-5 people to get a final dataset of multiple voices.

```bash
python3 src/0-get-data.py
```
<details>
<summary>What this script does:</summary>

- Prompts the user with recording instructions loaded from a JSON file.
- Records 60-second audio sessions with countdown and progress bar feedback.
- Splits each recording into overlapping 2s clips (50% overlap), trims silence, and discards too-short segments (max 79 segments per 1 min recording).
- Saves processed clips with unique IDs in a structured folder (processed_clips/speaker_id/).
- Generates a manifest CSV containing metadata (clip paths, speaker ID, script ID, timestamps).
- Entire generated dataset gets placed into `data/generated/`.

</details>
