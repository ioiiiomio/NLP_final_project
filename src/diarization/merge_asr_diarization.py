# src/diarization/merge_asr_diarization.py

import json
from pathlib import Path


def format_timestamp(t):
    """Convert float seconds → mm:ss with leading zeros."""
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_asr_and_diar(asr_json, diar_json):
    """
    Merge ASR segments (with word timestamps) and diarization segments.
    Produces human-readable transcript segments.
    """

    diar_segments = diar_json["segments"]
    output_segments = []

    for dseg in diar_segments:
        d_start = dseg["start"]
        d_end = dseg["end"]
        speaker = dseg["speaker"]

        collected_words = []

        # collect words from ASR that fall into this diarized segment
        for asr_seg in asr_json["segments"]:
            for w in asr_seg.get("words", []):
                if w["start"] >= d_start and w["end"] <= d_end:
                    collected_words.append(w["word"])

        text = " ".join(collected_words).strip()

        if text:
            output_segments.append({
                "speaker": speaker,
                "start": d_start,
                "end": d_end,
                "text": text
            })

    return output_segments


def save_readable_transcript(merged_segments, output_txt_path):
    """
    Save merged transcript in the required readable format:
    [00:00–00:17] SPEAKER_1: text...
    """
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for seg in merged_segments:
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            speaker = seg["speaker"].upper()
            text = seg["text"]

            f.write(f"[{ts_start}–{ts_end}] {speaker}: {text}\n")


def run_batch_merge():
    """
    For each audio file:
    - load ASR JSON from data/transcripts_diar/
    - load DIAR JSON from data/diarization_json/
    - merge
    - save human-readable transcript in data/merged_final/
    """

    asr_dir = Path("data/transcripts_diar")
    diar_dir = Path("data/diarization_json")
    output_dir = Path("data/merged_final")
    output_dir.mkdir(parents=True, exist_ok=True)

    asr_files = sorted(asr_dir.glob("*_whisperx_asr.json"))

    if not asr_files:
        print("No ASR JSON files found.")
        return

    for asr_file in asr_files:
        base = asr_file.stem.replace("_whisperx_asr", "")
        diar_file = diar_dir / f"{base}_diar.json"

        if not diar_file.exists():
            print(f"No diar JSON for {base}, skipping.")
            continue

        print(f"Merging: {base}")

        asr_json = load_json(asr_file)
        diar_json = load_json(diar_file)

        merged = merge_asr_and_diar(asr_json, diar_json)

        out_txt = output_dir / f"{base}_merged.txt"
        save_readable_transcript(merged, out_txt)

        print(f"Saved merged transcript → {out_txt}")


if __name__ == "__main__":
    # run_batch_merge()

    base = "meeting01"  # prefix of the file

    asr_file = Path(f"data/transcripts_diar/{base}_whisperx_asr.json")
    diar_file = Path(f"data/diarization_json/{base}_diar.json")

    print(f"Merging:\n  ASR:  {asr_file}\n  DIAR: {diar_file}")

    merged = merge_asr_and_diar(asr_file, diar_file)

    out_path = Path(f"data/merged_final/{base}_merged.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

  
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved merged output to {out_path}")

    #this produces the final data/merged_final/meeting01_merged.json