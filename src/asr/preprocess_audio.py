# This is the first file to work in ASR pipeline, we transform audio to 16kHz mono WAV

import librosa
import soundfile as sf
import os

processed_audio = "./processed_audio/"
def load_and_resample(audio_path, target_sr=16000):
    """
    Load audio and convert to target sampling rate.
    Returns: path to processed temp WAV file.
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    out_path = processed_audio.replace(".mp3", "_16k.wav").replace(".m4a", "_16k.wav")
    sf.write(out_path, audio, target_sr)
    return out_path
