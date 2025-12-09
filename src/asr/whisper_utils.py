# Useful functions for Whisper output
def print_segments(asr_json):
    for seg in asr_json["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        print(f"[{start:.2f} → {end:.2f}] {text}")

def extract_text(asr_json):
    return " ".join(seg["text"] for seg in asr_json["segments"])

def to_simple_segments(asr_json):
    """Return simplified list: [{'start':..., 'end':..., 'text':...}]"""
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in asr_json["segments"]
    ]
