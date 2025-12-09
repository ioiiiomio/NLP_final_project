# src/diarization/run_whisperx.py

import json
from pathlib import Path
import whisperx


def run_whisperx_on_file(audio_path, model, align_model, metadata, device="cpu"):
    """
    Runs WhisperX ASR + alignment on a single file.
    Returns aligned result (segments with word-level timestamps).
    """
    audio = whisperx.load_audio(str(audio_path))

    # Step 1: ASR
    asr_result = model.transcribe(audio)

    # Step 2: Alignment
    aligned = whisperx.align(
        asr_result["segments"],
        align_model,
        metadata,
        audio,
        device=device
    )

    return aligned


def run_batch_whisperx():
    device = "cpu"
    model_size = "medium.en"

    audio_dir = Path("data/raw_audio_diar")
    output_dir = Path("data/transcripts_diar")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WhisperX ASR model...")
    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type="int8",
        asr_backend="openai",
        vad_method="silero"
    )

    print("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model("en", device=device)

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print("No WAV files found!")
        return

    for audio_path in audio_files:
        print(f"Processing {audio_path.name}")

        aligned_result = run_whisperx_on_file(
            audio_path, model, align_model, metadata, device=device
        )

        out_file = output_dir / f"{audio_path.stem}_asr.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(aligned_result, f, indent=2, ensure_ascii=False)

        print(f"Saved ASR JSON → {out_file}")


if __name__ == "__main__":
    #run_batch_whisperx()

    audio_path = "data/raw_audio_diar/meeting01.wav"
    output_dir = Path("data/transcripts_diar")

    print(f"Running ASR for: {audio_path}")
    aligned_result = run_whisperx_on_file(
        Path(audio_path),
        whisperx.load_model("medium.en", device="cpu", compute_type="int8", asr_backend="openai"),
        *whisperx.load_align_model("en", device="cpu"),
        device="cpu"
    )

    out_file = output_dir / f"{audio_path.stem}_asr.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, indent=2, ensure_ascii=False)

    print(f"Saved ASR JSON → {out_file}")

    #this will produce data/transcripts_diar/meeting01_whisperx_asr.json


