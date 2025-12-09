from .run_whisper import run_whisper

def asr_pipeline(audio_path, model_size="medium"):
    """
    High-level ASR pipeline used by:
    - diarization module
    - notebooks
    - evaluation scripts
    """
    asr_json = run_whisper(audio_path, model_size=model_size)
    return asr_json
