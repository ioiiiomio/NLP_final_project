import json
import os
import argparse

from textrank import textrank_summarize
from abstractive_general import generate_general_summary
from abstractive_meeting import generate_meeting_summary


# ============================================================
# HELPERS
# ============================================================

def human_speaker(raw):
    """Convert SPEAKER_02 → Speaker 2."""
    if raw.upper().startswith("SPEAKER_"):
        num = raw.split("_")[1]
        if num.isdigit():
            return f"Speaker {int(num)}"
    return raw.replace("_", " ").title()


def load_segments(path):
    """
    Loads your exact input format:

    [
      {start, end, speaker, text, reactions: [...]},
      ...
    ]

    Flattens segments + reactions, sorts chronologically.
    """
    raw = json.load(open(path, "r", encoding="utf-8"))
    flat = []

    for seg in raw:
        flat.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg["speaker"],
            "text": seg["text"],
            "is_reaction": False
        })

        # Add reactions as separate small utterances
        for r in seg.get("reactions", []):
            flat.append({
                "start": r["start"],
                "end": r["end"],
                "speaker": r["speaker"],
                "text": r["text"],
                "is_reaction": True
            })

    # Sort by time
    flat.sort(key=lambda x: x["start"])
    return flat


def is_meeting(segments):
    """More than one speaker = meeting."""
    speakers = {seg["speaker"] for seg in segments}
    return len(speakers) > 1


# ============================================================
# BUILD FULL DIARIZED TRANSCRIPT
# ============================================================

def build_full_transcript(segments):
    """
    Produces clean, human-readable diarized dialogue:

    Speaker 2: Welcome...
    Speaker 0: I'm Gauhar...
    Speaker 3: I studied...
    """
    lines = []

    for seg in segments:
        if seg["is_reaction"]:
            continue  # reactions harm summarization
        sp = human_speaker(seg["speaker"])
        text = seg["text"].strip()
        lines.append(f"{sp}: {text}")

    return "\n".join(lines)


# ============================================================
# MINUTE CHUNKING (DIARIZED)
# ============================================================

def build_minute_chunks(segments):
    """
    Produces diarization-aware minute chunks such as:

    start: 00:00, end: 00:59
    text:
       Speaker 2: Welcome...
       Speaker 0: Absolutely...
       Speaker 0: My name is Gauhar...

    Reactions omitted.
    """
    chunks = {}

    for seg in segments:
        if seg["is_reaction"]:
            continue

        minute = int(seg["start"] // 60)
        sp = human_speaker(seg["speaker"])
        line = f"{sp}: {seg['text'].strip()}"

        chunks.setdefault(minute, [])
        chunks[minute].append(line)

    out = []
    for m, lines in sorted(chunks.items()):
        start = f"{m:02d}:00"
        end = f"{m:02d}:59"
        text_block = "\n".join(lines)
        out.append({"start": start, "end": end, "text": text_block})

    return out


def summarize_chunks(chunks):
    """Summarize each diarized minute chunk using the meeting model."""
    results = []
    for ch in chunks:
        payload = {"segments": [{"text": ch["text"]}]}
        summary = generate_meeting_summary(payload, max_length=180)

        results.append({
            "start": ch["start"],
            "end": ch["end"],
            "summary": summary
        })

    return results


# ============================================================
# SPEAKER SUMMARIES
# ============================================================

def speaker_summaries(segments):
    """
    Produces human-like summary per speaker.
    Reactions ignored.
    """
    sp_text = {}

    for seg in segments:
        if seg["is_reaction"]:
            continue
        sp = human_speaker(seg["speaker"])
        sp_text.setdefault(sp, "")
        sp_text[sp] += " " + seg["text"].strip()

    out = {}
    for sp, text in sp_text.items():
        payload = {"segments": [{"text": text}]}
        summary = generate_meeting_summary(payload, max_length=200)
        out[sp] = summary

    return out


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(input_path, output_dir="data/summaries"):
    os.makedirs(output_dir, exist_ok=True)

    segments = load_segments(input_path)
    meeting = is_meeting(segments)

    # ----------------- TEXTRANK (extractive) -----------------
    textrank_summary = textrank_summarize({"segments": segments}, 4)

    # ----------------- FULL ABSTRACTION -----------------------
    full_transcript = build_full_transcript(segments)
    payload = {"segments": [{"text": full_transcript}]}

    if meeting:
        abstractive_full = generate_meeting_summary(payload, max_length=260)
    else:
        abstractive_full = generate_general_summary(payload, max_length=260)

    # ----------------- MINUTE DIARIZATION CHUNKS --------------
    chunks = build_minute_chunks(segments)
    chunk_summaries = summarize_chunks(chunks)

    # ----------------- SPEAKER SUMMARIES ----------------------
    spk_sums = speaker_summaries(segments)

    # ----------------- OUTPUT JSON ----------------------------
    out = {
        "mode": "meeting" if meeting else "monologue",
        "textrank_summary": textrank_summary,
        "abstractive_full_summary": abstractive_full,
        "minute_chunks": chunk_summaries,
        "speaker_summaries": spk_sums
    }

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}_summary.json")

    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print(f"[OK] Saved summary to {out_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="data/summaries")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.output_dir)