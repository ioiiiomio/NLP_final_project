import json
import os
import argparse
import re
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

from textrank import textrank_summarize
from abstractive_general import (
    generate_general_summary,
    generate_general_summary_simple,
    analyze_speaker_pattern,
    detect_monologue_type
)
from abstractive_meeting import (
    generate_meeting_summary, 
    detect_conversation_type,
    MODEL_REGISTRY
)


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
    Loads your exact input format and flattens segments + reactions.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    flat = []
    
    for seg in raw:
        # Main segment
        flat.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg["speaker"],
            "text": seg["text"],
            "is_reaction": False
        })
        
        # Reactions
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
    speakers = {seg["speaker"] for seg in segments if not seg.get("is_reaction", False)}
    return len(speakers) > 1


# ============================================================
# BUILD FULL DIARIZED TRANSCRIPT (IMPROVED)
# ============================================================

def build_full_transcript(segments, include_reactions=False):
    """
    Produces clean, human-readable diarized dialogue.
    """
    lines = []
    
    for seg in segments:
        if not include_reactions and seg.get("is_reaction", False):
            continue
        
        sp = human_speaker(seg["speaker"])
        text = seg["text"].strip()
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        
        # Format based on whether it's a reaction
        if seg.get("is_reaction", False):
            lines.append(f"[{sp} reacts]: {text}")
        else:
            lines.append(f"{sp}: {text}")
    
    return "\n".join(lines)


def format_time(seconds):
    """Format seconds to MM:SS"""
    td = timedelta(seconds=int(seconds))
    mm, ss = divmod(td.seconds, 60)
    return f"{mm:02d}:{ss:02d}"


# ============================================================
# MINUTE CHUNKING (IMPROVED)
# ============================================================

def build_smart_chunks(segments, chunk_minutes=1):
    """
    Create smarter chunks based on conversation flow.
    """
    chunks = {}
    
    for seg in segments:
        if seg.get("is_reaction", False):
            continue
        
        # Determine chunk key (minute-based)
        minute = int(seg["start"] // 60)
        chunk_key = f"{minute:03d}"  # Pad with zeros for proper sorting
        
        # Get speaker and text
        sp = human_speaker(seg["speaker"])
        text = seg["text"].strip()
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        # Add to chunk
        if chunk_key not in chunks:
            chunks[chunk_key] = {
                "start_time": seg["start"],
                "end_time": seg["end"],
                "lines": [],
                "speakers": set()
            }
        
        chunks[chunk_key]["lines"].append(f"{sp}: {text}")
        chunks[chunk_key]["speakers"].add(sp)
        chunks[chunk_key]["end_time"] = max(chunks[chunk_key]["end_time"], seg["end"])
    
    # Convert to output format
    out = []
    for chunk_key, data in sorted(chunks.items()):
        start_min = int(data["start_time"] // 60)
        start_sec = int(data["start_time"] % 60)
        end_min = int(data["end_time"] // 60)
        end_sec = int(data["end_time"] % 60)
        
        # Calculate chunk duration
        duration = data["end_time"] - data["start_time"]
        
        out.append({
            "chunk_id": f"chunk_{chunk_key}",
            "start": f"{start_min:02d}:{start_sec:02d}",
            "end": f"{end_min:02d}:{end_sec:02d}",
            "duration_seconds": round(duration, 2),
            "text": "\n".join(data["lines"]),
            "speakers": list(data["speakers"]),
            "line_count": len(data["lines"])
        })
    
    return out


def summarize_chunks(chunks, model_type="auto", use_ml_detection=False):
    """Summarize each chunk using appropriate model."""
    results = []
    
    for i, ch in enumerate(chunks):
        # Create payload for this chunk
        payload = {"segments": [{"text": ch["text"]}]}
        
        # Generate summary
        try:
            result = generate_meeting_summary(
                payload, 
                model_type=model_type,
                max_length=120,  # Shorter for chunks
                force_diarization=True,
                use_ml_detection=use_ml_detection
            )
            
            results.append({
                "chunk_index": i,
                "chunk_id": ch.get("chunk_id", f"chunk_{i}"),
                "start": ch["start"],
                "end": ch["end"],
                "duration_seconds": ch.get("duration_seconds", 0),
                "summary": clean_chunk_summary(result["summary"]),
                "model_used": result["model_used"],
                "model_type": result["model_type"],
                "speakers": ch.get("speakers", []),
                "line_count": ch.get("line_count", 0)
            })
        except Exception as e:
            print(f"[WARNING] Failed to summarize chunk {i}: {e}")
            results.append({
                "chunk_index": i,
                "chunk_id": ch.get("chunk_id", f"chunk_{i}"),
                "start": ch["start"],
                "end": ch["end"],
                "summary": f"Summary unavailable for this segment.",
                "model_used": "error",
                "error": str(e)
            })
    
    return results


# ============================================================
# SPEAKER SUMMARIES (IMPROVED)
# ============================================================

def speaker_summaries(segments, model_type="auto", use_ml_detection=False):
    """
    Produces high-quality summary per speaker.
    """
    # Group text by speaker
    sp_text = defaultdict(list)
    sp_segments = defaultdict(list)
    
    for seg in segments:
        if seg.get("is_reaction", False):
            continue
        
        sp = human_speaker(seg["speaker"])
        text = seg["text"].strip()
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        sp_text[sp].append(text)
        sp_segments[sp].append(seg)
    
    # Generate summaries
    out = {}
    for sp, texts in sp_text.items():
        # Join texts with context
        full_text = " ".join(texts)
        
        # Create payload
        payload = {"segments": [{"text": f"{sp}: {full_text}"}]}
        
        # Generate summary
        try:
            result = generate_meeting_summary(
                payload,
                model_type=model_type,
                max_length=150,
                force_diarization=True,
                use_ml_detection=use_ml_detection
            )
            
            # Calculate statistics for this speaker
            speaker_segments = sp_segments[sp]
            word_count = sum(len(text.split()) for text in texts)
            avg_words = word_count / len(texts) if texts else 0
            
            out[sp] = {
                "summary": clean_speaker_summary(result["summary"], sp),
                "model_used": result["model_used"],
                "model_type": result["model_type"],
                "utterance_count": len(texts),
                "word_count": word_count,
                "avg_words_per_utterance": round(avg_words, 1),
                "time_range": {
                    "start": format_time(speaker_segments[0]["start"]) if speaker_segments else "00:00",
                    "end": format_time(speaker_segments[-1]["end"]) if speaker_segments else "00:00"
                }
            }
        except Exception as e:
            print(f"[WARNING] Failed to generate summary for {sp}: {e}")
            out[sp] = {
                "summary": f"Summary unavailable for {sp}.",
                "error": str(e),
                "utterance_count": len(texts)
            }
    
    return out


def clean_speaker_summary(summary, speaker_name):
    """Clean speaker-specific summaries"""
    if not summary:
        return ""
    
    # Remove repetitive speaker mentions
    pattern = rf"{re.escape(speaker_name)}:\s*"
    summary = re.sub(pattern, '', summary)
    
    # Remove common prefixes
    prefixes = [
        "the speaker says",
        "the speaker explains", 
        "according to the speaker",
        f"{speaker_name} says",
        f"{speaker_name} explains"
    ]
    
    for prefix in prefixes:
        if summary.lower().startswith(prefix):
            summary = summary[len(prefix):].strip()
            break
    
    # Ensure it starts clean
    summary = summary.strip()
    if summary and not summary[0].isupper():
        summary = summary[0].upper() + summary[1:]
    
    # Ensure it ends with period
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(input_path, output_dir="data/summaries", model_type="auto", 
                 use_ensemble=False, use_ml_detection=False, chunk_size=2):
    """
    Main pipeline with improved output quality.
    
    Args:
        input_path: Path to input JSON transcript
        output_dir: Output directory for summaries
        model_type: 'auto' (detect), 'interview', 'social', 'meeting', 'general'
        use_ensemble: Generate summaries with all models for comparison
        use_ml_detection: Use ML-based conversation type detection
        chunk_size: Size of time chunks in minutes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process segments
    print(f"[INFO] Loading transcript: {input_path}")
    segments = load_segments(input_path)
    
    if not segments:
        print("[ERROR] No segments found in input file")
        return None
    
    # Analyze speaker pattern to determine if it's truly a monologue
    speaker_analysis = analyze_speaker_pattern(segments)
    is_meeting_convo = not speaker_analysis["is_monologue"]
    confidence = speaker_analysis.get("confidence", 0.5)
    
    print(f"[INFO] Meeting conversation: {is_meeting_convo} (confidence: {confidence:.2f})")
    print(f"[INFO] Total segments: {len(segments)}")
    print(f"[INFO] Unique speakers: {len({seg['speaker'] for seg in segments if not seg.get('is_reaction', False)})}")
    
    # Build full transcript for detection
    full_transcript = build_full_transcript(segments, include_reactions=False)
    detection_payload = {"segments": [{"text": full_transcript}]}
    
    # Determine model type
    if model_type == "auto":
        print("[INFO] Auto-detecting conversation type...")
        if is_meeting_convo:
            # Use meeting detection
            detected_type = detect_conversation_type(detection_payload, use_ml=use_ml_detection)
        else:
            # Use monologue detection
            raw_text = " ".join([seg["text"] for seg in segments if not seg.get("is_reaction", False)])
            detected_type = "general"  # Use general for monologues
            content_type = detect_monologue_type(raw_text)
            print(f"[INFO] Monologue content type: {content_type}")
        
        print(f"[INFO] Detected type: {detected_type}")
        if detected_type in MODEL_REGISTRY:
            print(f"[INFO] Using model: {MODEL_REGISTRY[detected_type]['model_name']}")
    else:
        detected_type = model_type
        print(f"[INFO] Using forced model type: {detected_type}")
    
    # ----------------- TEXTRANK (extractive fallback) -----------------
    print("[INFO] Generating TextRank extractive summary...")
    try:
        textrank_result = textrank_summarize({"segments": segments}, 4)
        textrank_clean = clean_textrank_output(textrank_result)
    except Exception as e:
        print(f"[WARNING] TextRank failed: {e}")
        textrank_clean = "TextRank summary unavailable."
    
    # ----------------- FULL ABSTRACTIVE SUMMARY -----------------------
    print("[INFO] Generating full abstractive summary...")
    full_payload = {"segments": [{"text": full_transcript}]}
    
    if is_meeting_convo:
        try:
            abstractive_result = generate_meeting_summary(
                full_payload,
                model_type=detected_type,
                max_length=300,
                force_diarization=True,
                use_ml_detection=use_ml_detection
            )
            abstractive_full = abstractive_result["summary"]
            model_used = abstractive_result["model_used"]
            model_type_used = abstractive_result["model_type"]
        except Exception as e:
            print(f"[ERROR] Meeting summary generation failed: {e}")
            # Fallback to general summary
            abstractive_full = generate_general_summary_simple(full_payload, max_length=250)
            model_used = "facebook/bart-large-cnn (fallback)"
            model_type_used = "general"
    else:
        # Monologue - use general summary
        try:
            abstractive_full = generate_general_summary_simple(full_payload, max_length=250)
            model_used = "facebook/bart-large-cnn"
            model_type_used = "general"
        except Exception as e:
            print(f"[ERROR] Monologue summary generation failed: {e}")
            abstractive_full = "Summary generation failed."
            model_used = "error"
            model_type_used = "error"
    
    # Clean abstractive summary
    abstractive_clean = clean_abstractive_summary(abstractive_full, is_meeting_convo)
    
    # ----------------- SMART CHUNK SUMMARIES --------------------------
    print(f"[INFO] Generating {chunk_size}-minute chunk summaries...")
    try:
        chunks = build_smart_chunks(segments, chunk_minutes=chunk_size)
        chunk_summaries = summarize_chunks(
            chunks, 
            model_type=detected_type if is_meeting_convo else "general",
            use_ml_detection=use_ml_detection
        )
    except Exception as e:
        print(f"[WARNING] Chunk summarization failed: {e}")
        chunk_summaries = []
    
    # ----------------- SPEAKER SUMMARIES ------------------------------
    print("[INFO] Generating speaker summaries...")
    if is_meeting_convo:
        try:
            speaker_sums = speaker_summaries(
                segments, 
                model_type=detected_type,
                use_ml_detection=use_ml_detection
            )
        except Exception as e:
            print(f"[WARNING] Speaker summarization failed: {e}")
            speaker_sums = {}
    else:
        # For monologues, create a single speaker summary
        main_speaker = human_speaker(segments[0]["speaker"]) if segments else "Speaker"
        utterance_count = len([s for s in segments if not s.get("is_reaction", False)])
        
        speaker_sums = {
            main_speaker: {
                "summary": abstractive_clean,
                "model_used": model_used,
                "model_type": model_type_used,
                "utterance_count": utterance_count,
                "word_count": sum(len(seg["text"].split()) for seg in segments if not seg.get("is_reaction", False)),
                "avg_words_per_utterance": round(sum(len(seg["text"].split()) for seg in segments if not seg.get("is_reaction", False)) / max(utterance_count, 1), 1),
                "time_range": {
                    "start": format_time(segments[0]["start"]) if segments else "00:00",
                    "end": format_time(segments[-1]["end"]) if segments else "00:00"
                }
            }
        }
    
    # ----------------- CONVERSATION ANALYSIS --------------------------
    print("[INFO] Analyzing conversation characteristics...")
    try:
        conv_analysis = analyze_conversation_detailed(segments)
    except Exception as e:
        print(f"[WARNING] Conversation analysis failed: {e}")
        conv_analysis = {"error": str(e)}
    
    # ----------------- ENSEMBLE OPTION --------------------------------
    ensemble_results = None
    if use_ensemble:
        print("[INFO] Generating ensemble summaries...")
        try:
            ensemble_results = generate_ensemble_summaries(full_payload, use_ml_detection)
        except Exception as e:
            print(f"[WARNING] Ensemble generation failed: {e}")
            ensemble_results = {"error": str(e)}
    
    # ----------------- OUTPUT JSON ------------------------------------
    out = {
        "metadata": {
            "input_file": os.path.basename(input_path),
            "processing_time": get_current_timestamp(),
            "total_segments": len(segments),
            "reaction_segments": len([s for s in segments if s.get("is_reaction", False)]),
            "total_speakers": len({seg["speaker"] for seg in segments if not seg.get("is_reaction", False)}),
            "duration_seconds": segments[-1]["end"] if segments else 0,
            "duration_formatted": format_time(segments[-1]["end"]) if segments else "00:00",
            "conversation_type": detected_type,
            "is_meeting": is_meeting_convo,
            "monologue_confidence": round(confidence, 3),
            "model_type_used": model_type_used,
            "chunk_size_minutes": chunk_size
        },
        "mode": "meeting" if is_meeting_convo else "monologue",
        "summaries": {
            "textrank_extractive": textrank_clean,
            "abstractive_full": abstractive_clean,
            "abstractive_model": model_used
        },
        "temporal_analysis": {
            "minute_chunks": chunk_summaries,
            "total_chunks": len(chunk_summaries)
        },
        "speaker_analysis": speaker_sums,
        "conversation_analysis": conv_analysis
    }
    
    if ensemble_results:
        out["ensemble_comparison"] = ensemble_results
    
    # Save output
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}_summary.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Saved summary to {out_path}")
    print(f"[INFO] Conversation type: {detected_type}")
    print(f"[INFO] Model used: {model_used}")
    print(f"[INFO] Summary length: {len(abstractive_clean.split())} words")
    
    return out


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_textrank_output(text):
    """Clean TextRank extractive summary"""
    if not text:
        return "No summary generated."
    
    # Remove duplicate sentences
    sentences = text.split('. ')
    unique_sentences = []
    seen = set()
    
    for sent in sentences:
        sent_clean = sent.strip()
        if sent_clean and sent_clean not in seen:
            seen.add(sent_clean)
            unique_sentences.append(sent_clean)
    
    # Join and ensure proper punctuation
    result = '. '.join(unique_sentences)
    if result and not result.endswith('.'):
        result += '.'
    
    # Remove very short summaries
    if len(result.split()) < 5:
        return "TextRank produced very short summary. Using abstractive only."
    
    return result


def clean_abstractive_summary(summary, is_meeting):
    """Clean abstractive summary"""
    if not summary:
        return "No summary generated."
    
    # Remove incomplete sentences at beginning/end
    summary = re.sub(r'^[^A-Za-z]*', '', summary)
    
    # Ensure it starts properly
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    # For meetings, ensure speaker references are clean
    if is_meeting:
        summary = re.sub(r'(Speaker \d+: ){2,}', 'Speaker: ', summary)
        summary = re.sub(r'^(The (speaker|participant|person) (says|explains|states|mentions) (that )?)', '', summary, flags=re.IGNORECASE)
    
    # Fix spacing
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # Ensure it ends with period
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    # Remove empty summary
    if len(summary.split()) < 3:
        return "Summary too short. May indicate processing issues."
    
    return summary


def clean_chunk_summary(summary):
    """Clean chunk summaries"""
    if not summary:
        return "No summary for this segment."
    
    # Remove any "The speaker says" prefixes
    summary = re.sub(r'^(The (speaker|host|guest|participant) (says|explains|mentions|states|adds|notes|asks|tells) that?\s*)?', 
                    '', summary, flags=re.IGNORECASE)
    
    # Capitalize if needed
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    # Ensure proper ending
    summary = summary.strip()
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary


def analyze_conversation_detailed(segments):
    """Analyze conversation characteristics in detail"""
    speakers = defaultdict(lambda: {"count": 0, "words": 0, "questions": 0})
    utterances = 0
    total_words = 0
    questions = 0
    reactions = 0
    
    for seg in segments:
        is_reaction = seg.get("is_reaction", False)
        speaker = seg["speaker"]
        text = seg["text"]
        
        if is_reaction:
            reactions += 1
            continue
        
        # Update speaker stats
        speakers[speaker]["count"] += 1
        words = len(text.split())
        speakers[speaker]["words"] += words
        total_words += words
        
        if "?" in text:
            speakers[speaker]["questions"] += 1
            questions += 1
        
        utterances += 1
    
    # Calculate statistics
    unique_speakers = len(speakers)
    avg_utterance_length = total_words / max(utterances, 1)
    
    # Find dominant speaker
    speaker_stats = []
    for sp, stats in speakers.items():
        speaker_stats.append({
            "speaker": human_speaker(sp),
            "utterances": stats["count"],
            "words": stats["words"],
            "questions": stats["questions"],
            "avg_words": stats["words"] / max(stats["count"], 1),
            "percentage": (stats["count"] / max(utterances, 1)) * 100
        })
    
    # Sort by utterance count
    speaker_stats.sort(key=lambda x: x["utterances"], reverse=True)
    
    return {
        "speaker_count": unique_speakers,
        "total_utterances": utterances,
        "total_words": total_words,
        "reaction_count": reactions,
        "question_count": questions,
        "avg_utterance_length": round(avg_utterance_length, 1),
        "speaker_distribution": speaker_stats,
        "dominant_speaker": speaker_stats[0]["speaker"] if speaker_stats else "None",
        "dominant_percentage": round(speaker_stats[0]["percentage"], 1) if speaker_stats else 0
    }


def generate_ensemble_summaries(payload, use_ml_detection=False):
    """Generate summaries with all models for comparison"""
    results = {}
    
    for model_type, config in MODEL_REGISTRY.items():
        try:
            # Use the meeting summary function but force model type
            result = generate_meeting_summary(
                payload,
                model_type=model_type,
                max_length=250,
                force_diarization=True,
                use_ml_detection=use_ml_detection
            )
            results[model_type] = {
                "summary": result["summary"],
                "model": config["model_name"],
                "description": config["description"],
                "length_words": len(result["summary"].split())
            }
        except Exception as e:
            results[model_type] = {
                "error": str(e),
                "model": config["model_name"],
                "description": config["description"]
            }
    
    return results


def get_current_timestamp():
    """Get current timestamp for metadata"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced conversation summarization pipeline with multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input transcript.json
  %(prog)s --input transcript.json --model_type interview
  %(prog)s --input transcript.json --use_ml --ensemble
  %(prog)s --input transcript.json --chunk_size 1 --output_dir ./summaries
        """
    )
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to input JSON transcript"
    )
    parser.add_argument(
        "--output_dir", 
        default="data/summaries",
        help="Output directory for summaries (default: data/summaries)"
    )
    parser.add_argument(
        "--model_type", 
        default="auto",
        choices=["auto", "interview", "social", "meeting", "general"],
        help="Conversation type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--use_ml",
        action="store_true",
        help="Use ML-based conversation type detection (if available)"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Generate ensemble summaries with all models for comparison"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2,
        choices=[1, 2, 3, 5, 10],
        help="Chunk size in minutes (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set verbosity
    if not args.verbose:
        # Reduce output noise
        warnings.filterwarnings("ignore")
    
    try:
        result = run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            model_type=args.model_type,
            use_ensemble=args.ensemble,
            use_ml_detection=args.use_ml,
            chunk_size=args.chunk_size
        )
        
        if result:
            print("\n" + "="*60)
            print("SUMMARY GENERATION COMPLETE")
            print("="*60)
            print(f"Input file: {args.input}")
            print(f"Output file: {args.output_dir}/{os.path.basename(args.input).replace('.json', '_summary.json')}")
            print(f"Mode: {result.get('mode', 'unknown')}")
            print(f"Model used: {result.get('summaries', {}).get('abstractive_model', 'unknown')}")
            
            # Print a preview of the summary
            summary = result.get('summaries', {}).get('abstractive_full', '')
            if summary:
                print("\nPreview of abstractive summary:")
                print("-"*40)
                print(summary[:200] + "..." if len(summary) > 200 else summary)
                print("-"*40)
                
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON file: {e}")
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()