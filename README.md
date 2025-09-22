# speaker-recognition

A machine learning pipeline for classifying a speaker from their voice.

---

## Project Structure

<!-- ```
genre-classification/
│
├── data/                   # Dataset lives here (GTZAN)
│ └── gtzan/
│   └── genres/             # 10 genre folders with .wav files
│
├── notebooks/              # Jupyter notebooks for EDA and experiments
│
├── scripts/                # Utility scripts
│
├── src/                    # Source code
│ └── 0-get-data.py         # Downloads GTZAN dataset into data/gtzan
│ └── 1-extract-features.py # Extracts features from the GTZAN dataset
│
├── README.md
├── requirements.txt
└── .gitignore
``` -->

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

<!-- ### 4. Download the dataset

```bash
python3 scripts/download_data.py
```
This will place the dataset into `data/gtzan/genres/`. -->