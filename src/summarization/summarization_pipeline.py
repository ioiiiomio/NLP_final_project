import json
import os
import argparse

from textrank import textrank_summarize
from abstractive_meeting import generate_meeting_summary, generate_meeting_summary_simple, get_conversation_analysis
from abstractive_general import generate_general_summary



def human_speaker(raw):
    if raw.upper().startswith("SPEAKER_"):
        num = raw.split("_")[1]
        if num.isdigit():
            return f"Speaker {int(num)}"
    return raw.replace("_", " ").title()

def load_segments(path):
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

        for r in seg.get("reactions", []):
            flat.append({
                "start": r["start"],
                "end": r["end"],
                "speaker": r["speaker"],
                "text": r["text"],
                "is_reaction": True
            })

    flat.sort(key=lambda x: x["start"])
    return flat

def is_meeting(segments):
    speakers = {seg["speaker"] for seg in segments if not seg.get("is_reaction", False)}
    return len(speakers) > 1


def build_full_transcript(segments):
    lines = []

    for seg in segments:
        if seg.get("is_reaction", False):
            continue  
        sp = human_speaker(seg["speaker"])
        text = seg["text"].strip()
        lines.append(f"{sp}: {text}")

    return "\n".join(lines)


def build_minute_chunks(segments):
    chunks = {}

    for seg in segments:
        if seg.get("is_reaction", False):
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

def summarize_chunks(chunks, conversation_type="auto", use_enhanced=True):
    results = []
    
    for ch in chunks:
        payload = {"segments": [{"text": ch["text"]}]}
        
        if use_enhanced:
            result = generate_meeting_summary(
                payload, 
                max_length=180,
                model_type=conversation_type,
                force_diarization=True
            )
            summary = result["summary"]
        else:
            summary = generate_meeting_summary_simple(payload, max_length=180)
        
        results.append({
            "start": ch["start"],
            "end": ch["end"],
            "summary": summary
        })
    
    return results

def run_pipeline(input_path, output_dir="data/summaries", use_enhanced=True):
    os.makedirs(output_dir, exist_ok=True)

    segments = load_segments(input_path)
    meeting = is_meeting(segments)
    
    conversation_type = "auto"
    if use_enhanced:
        conv_analysis = get_conversation_analysis({"segments": segments})
        conversation_type = conv_analysis.get("detected_type", "auto")
        print(f"[INFO] Detected: {conversation_type}")
        print(f"[INFO] Speakers: {conv_analysis.get('unique_speakers', [])}")
    
    print("[INFO] Generating TextRank extractive summary...")
    textrank_summary = textrank_summarize({"segments": segments}, 4)

    print("[INFO] Generating full abstractive summary...")
    full_transcript = build_full_transcript(segments)
    payload = {"segments": [{"text": full_transcript}]}

    if meeting:
        if use_enhanced:
            result = generate_meeting_summary(
                payload, 
                max_length=260,
                model_type=conversation_type,
                force_diarization=True
            )
            abstractive_full = result["summary"]
            model_used = result["model_used"]
        else:
            abstractive_full = generate_meeting_summary_simple(payload, max_length=260)
            model_used = "mikeadimech/longformer-qmsum-meeting-summarization"
    else:
        abstractive_full = generate_general_summary(payload, max_length=260)
        model_used = "facebook/bart-large-cnn"


    print("[INFO] Generating minute chunk summaries...")
    chunks = build_minute_chunks(segments)
    chunk_summaries = summarize_chunks(chunks, conversation_type, use_enhanced)

    out = {
        "mode": "meeting" if meeting else "monologue",
        "conversation_type": conversation_type if use_enhanced else ("meeting" if meeting else "monologue"),
        "model_used": model_used,
        "textrank_summary": textrank_summary,
        "abstractive_full_summary": abstractive_full,
        "minute_chunks": chunk_summaries
    }

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}_summary.json")

    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print(f"[OK] Saved summary to {out_path}") 



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", default="data/summaries", help="Output directory for summaries")
    parser.add_argument("--legacy", action="store_true", help="Use legacy model selection (simple meeting/monologue)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    use_enhanced = not args.legacy
    
    if use_enhanced:
        print("[INFO] Running in ENHANCED mode with specialized models")
    else:
        print("[INFO] Running in LEGACY mode with simple meeting/monologue detection")
    

    os.environ["HF_HUB_OFFLINE"] = "1"
    
    run_pipeline(args.input, args.output_dir, use_enhanced)