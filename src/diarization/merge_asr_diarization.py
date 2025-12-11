import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def time_overlap(a_start, a_end, b_start, b_end):
    """Compute overlap between two time intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def assign_speaker(asr_seg, diar_segments):
    """
    Assign the speaker with the largest time overlap with this ASR segment.
    """
    a_start = asr_seg["start"]
    a_end = asr_seg["end"]

    overlaps = {}

    for diar in diar_segments:
        d_start = diar["start"]
        d_end = diar["end"]
        overlap = time_overlap(a_start, a_end, d_start, d_end)

        if overlap > 0:
            speaker = diar["speaker"]
            overlaps[speaker] = overlaps.get(speaker, 0) + overlap

    if not overlaps:
        return "UNKNOWN"

    # Return the speaker with the maximum overlap
    return max(overlaps, key=overlaps.get)


def merge_asr_and_diar(asr_json, diar_json):
    asr_segments = asr_json["segments"]
    diar_segments = diar_json["segments"]

    merged = []

    for asr_seg in asr_segments:
        speaker = assign_speaker(asr_seg, diar_segments)

        merged.append({
            "start": asr_seg["start"],
            "end": asr_seg["end"],
            "speaker": speaker,
            "text": asr_seg["text"],
            "words": asr_seg.get("words", [])
        })

    return merged


