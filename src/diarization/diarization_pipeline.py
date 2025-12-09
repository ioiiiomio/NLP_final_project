# src/diarization/diarization_pipeline.py

import json
from pathlib import Path
import whisperx


def run_diarization_on_file(audio_path, diarize_model, device="cpu"):
    """
    Run WhisperX diarization on a single audio file.
    Returns standardized diarization segment list.
    """
    result = diarize_model(audio_path)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "speaker": seg["speaker"],
            "start": float(seg["start"]),
            "end": float(seg["end"])
        })

    return segments


def run_batch_diarization():
    """
    Processes all WAV files inside data/raw_audio_diar/
    and stores diarization JSONs inside data/diarization_json/
    """
    device = "cpu"

    audio_dir = Path("data/raw_audio_diar")
    output_dir = Path("data/diarization_json")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WhisperX diarization model...")
    diarize_model = whisperx.load_diarize_model("en", device=device)

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"No .wav files found inside: {audio_dir}")
        return

    for audio_path in audio_files:
        print(f"\nDiarizing: {audio_path.name}")

        diar_segments = run_diarization_on_file(
            str(audio_path),
            diarize_model,
            device=device
        )

        output_data = {"segments": diar_segments}
        out_file = output_dir / f"{audio_path.stem}_diar.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved diarization JSON → {out_file}")



if __name__ == "__main__":
    #run_batch_diarization()

    audio_path = "data/raw_audio_diar/meeting01.wav"
    output_dir = Path("data/diarization_json")

    print(f"Running diarization for: {audio_path}")
    diar_segm = run_diarization_on_file(audio_path)

    output_data = {"segments": diar_segm}
    out_file = output_dir / f"{audio_path.stem}_diar.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved diarization JSON → {out_file}")

    #this will produce data/diarization_json/meeting01_diar.json