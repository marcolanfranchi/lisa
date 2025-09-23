# speaker-recognition

A machine learning pipeline for classifying a speaker from their voice.

---

## Project Structure

```
speaker-recognition/
│
├── data/                       # Data
│ └── generated/                # Generated dataset of voice recordings
│ └── recording-prompts.json    # Voice recording instructions
│
├── src/                        # Source code
│ └── 0-get-data.py             # Starts the data generation process
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
This will place the dataset into `data/generated/`.