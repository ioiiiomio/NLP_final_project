### Speech-to-Text Summarization System

This project implements an end-to-end pipeline for converting speech to text and generating summaries from the transcript.

The system supports:

- ASR (Automatic Speech Recognition) using Whisper
- Diarization using WhisperX (speaker detection)
- Summarization using T5-small or DistilBART

### Two summary modes:
- General summary

- Speaker-specific summary

**Specifics: For each summary modes, Timestamp-based summarization (summaries for only selected segments)**

The repository is organized into modular components to allow each team member to work independently and integrate seamlessly.

### Repository structure: 
```final_project/
    │
    ├── data/
    │   ├── raw_audio/                 # Original audio files
    │   ├── transcripts_asr/           # Whisper transcription outputs (timestamps)
    │   ├── transcripts_diarized/      # WhisperX diarized transcripts (speaker labels)
    │   ├── final_json/                # Unified JSON format used for summarization
    │   └── gold/                      # Manually created transcripts/summaries for evaluation
    │
    ├── src/
    │   ├── asr/                       # ASR module (Person 1)
    │   │   ├── preprocess_audio.py        # Audio loading, resampling, cleaning
    │   │   ├── run_whisper.py             # Whisper transcription runner
    │   │   ├── whisper_utils.py           # Helper functions for Whisper output handling
    │   │   └── asr_pipeline.py            # Full ASR pipeline (importable by other modules)
    │   │
    │   ├── diarization/               # Diarization module (Person 2)
    │   │   ├── run_whisperx.py             # WhisperX diarization scripts
    │   │   ├── merge_asr_diarization.py    # Align ASR timestamps with speaker info
    │   │   └── diarization_pipeline.py     # Unified diarization pipeline
    │   │
    │   ├── summarization/             # Summarization module (Person 3)
    │   │   ├── textrank.py                 # Extractive summarization baseline (TextRank)
    │   │   ├── abstractive_t5.py           # T5-small summarizer
    │   │   ├── abstractive_distilbart.py   # DistilBART summarizer
    │   │   └── summarization_pipeline.py   # Main summarization orchestrator
    │   │
    │   ├── utils/                     # Shared helper utilities
    │   │   ├── file_io.py
    │   │   ├── json_formatter.py
    │   │   └── chunking.py
    │   │
    │   └── pipeline.py                # End-to-end pipeline combining all components
    │
    ├── notebooks/
    │   ├── 01_ASR_demo.ipynb              # Demo notebook for ASR module
    │   ├── 02_Diarization_demo.ipynb      # Demo notebook for diarization
    │   ├── 03_Summarization_demo.ipynb    # Demo notebook for summarization
    │   ├── 04_E2E_demo.ipynb              # End-to-end demonstration
    │   └── 05_Evaluation.ipynb            # ROUGE evaluation & comparisons
    │
    ├── tests/                            # Unit tests for each module
    │   ├── test_asr.py
    │   ├── test_diarization.py
    │   ├── test_summarization.py
    │   └── test_pipeline.py
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

*** ! You can Adjust your specific folders, those are just a template. But, try not to break the structure too much, please ;> ! ***

### How the components work together : 
```Audio File
    │
    ▼
    ASR (Whisper) 
    │  produces timestamps
    ▼
    Diarization (WhisperX)
    │  assigns speakers
    ▼
    Final JSON 
    │
    ├── General Summarizer (T5/DistilBART)
    └── Speaker-Specific Summarizer

### Full description of each folder(Please, update if you change anything):
*** data/ ***

- All data files are organized here to keep the code clean.
- raw_audio/ — audio files used for testing and demos
- transcripts_asr/ — outputs from Whisper (timestamped segments)
- transcripts_diarized/ — outputs from WhisperX (speaker-labeled segments)
- final_json/ — merged ASR + diarization files used as the final input to summarization
- gold/ — optional: hand-made transcripts and summaries for evaluation (ROUGE metrics)

### Setup instructions
1) Clone repository
2) reate virtual environment
    python3 -m venv venv
    source venv/bin/activate
3) Install dependencies: 
    pip install -r requirements.txt

### Contribution Guide
#### Branches(create your own when you work, don't push to main!!! NEVERRR!!!):
asr-dev -> zhaniya
diarization-dev -> akzhan
summarization-dev -> inkar
integration -> zhaniya
main -> zhaniya

# NEVER WRITE TOKENS, AND OTHER SENSITIVE DATA EXPLICITLY! ADD THE TOKENS TO THE .env file