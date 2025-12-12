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
```
final_project/
│
├── data/
│   ├── asr_json/                 
│   ├── diarized_json/          
│   ├── final_json/      
│   ├── processed_audio/            
│   ├── pure_merge/                
│   └── raw_audio/                
│   └── semantic_merge/           
│
├── src/
│   ├── asr/                       # ASR module (Zhaniya)
│   │   ├── asr_pipeline.ipynb        
│   │   ├── run_asr.py             # ASR runner
│   │   └── asr_pipeline.py        # Full ASR pipeline from netebook to py (importable by other modules)
│   │
│   ├── diarization/               # Diarization module (Akzhan)
│   │   ├── diarization.py             
│   │   ├── merge_asr_diarization.py    
│   │   └── semantic_merge_pipeline.py     
│   │
│   ├── summarization/             # Summarization module (Inkar)
│   │   ├── textrank.py                 
│   │   ├── abstractive_t5.py           
│   │   ├── abstractive_distilbart.py   
│   │   └── summarization_pipeline.py   
│
├── notebooks/
│   ├── 01_ASR_demo.ipynb              
│   ├── 02_Diarization_demo.ipynb      
│   ├── 03_Summarization_demo.ipynb    
│   ├── 04_E2E_demo.ipynb              
│   └── 05_Evaluation.ipynb            
├── requirements.txt
├── .gitignore
└── README.md
```

*** ! You can Adjust your specific folders, those are just a template. But, try not to break the structure too much, please ;> ! ***

### How the components work together : 
```
Audio File
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
```

### Full description of each folder(Please, update after you write your part):
*** data/ ***

- All data files are organized here to keep the code clean.
- raw_audio/ — audio files used for testing and demos
- transcripts_asr/ — outputs from Whisper (timestamped segments)
- transcripts_diarized/ — outputs from WhisperX (speaker-labeled segments)
- final_json/ — merged ASR + diarization files used as the final input to summarization
- gold/ — optional: hand-made transcripts and summaries for evaluation (ROUGE metrics)

### Setup instructions
1) Clone repository
2) Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
3) Install dependencies(update the file if your code context created new dependencies): 
    pip install -r requirements.txt

### Contribution Guide
#### Branches(create your own when you work, don't push to main!!! NEVERRR!!!):
asr-dev -> zhaniya
diarization-dev -> akzhan
summarization-dev -> inkar
tests -> everyone
integration -> zhaniya
main -> zhaniya

# NEVER WRITE TOKENS, AND OTHER SENSITIVE DATA EXPLICITLY! ADD THE TOKENS TO THE .env file