# src/asr/run_whisper.py

import argparse
import json
import os

import whisperx
from tqdm import tqdm


def run_whisper(audio_path, model_size="medium.en", output_dir="data/transcripts_asr"):
    device = "cpu"

    print("Loading WhisperX model...")
    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type="int8",
        vad_method = "silero"  
    )


    print("Transcribing...")
    asr_result = model.transcribe(audio_path)

    for _ in tqdm(asr_result["segments"], desc="ASR Segments"):
        pass

    print("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model(asr_result["language"], device=device)

    print("Running alignment...")
    aligned = whisperx.align(
        asr_result["segments"],
        align_model,
        metadata,
        audio_path,
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(audio_path))[0] + ".json"
    out_path = os.path.join(output_dir, out_name)

    with open(out_path, "w") as f:
        json.dump(aligned, f, indent=2)

    print(f"Saved ASR transcript to: {out_path}")
    return aligned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True)
    args = parser.parse_args()

    run_whisper(args.audio_path)
