import json 
# 1. Load ASR and Diarization JSON files
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# 2. Match speaker label to each ASR segment
def assign_speakers(asr_segments, diar_segments):
    """
    For each ASR segment, find the diarization segment
    that overlaps the most and assign its speaker label.
    """
    for seg in asr_segments:
        s_start, s_end = seg["start"], seg["end"]
        best_overlap = 0
        best_speaker = "UNKNOWN"

        for d in diar_segments:
            d_start, d_end = d["start"], d["end"]
            overlap = max(0, min(s_end, d_end) - max(s_start, d_start))

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        seg["speaker"] = best_speaker

    return asr_segments

# 3. Semantic floor-taking merge

def merge_floor_taking(asr_segments, reaction_threshold=3.0):
    """
    reaction_threshold: max duration (in seconds) to consider something a reaction
    """

    merged = []
    current = None  # current main speaker block

    for seg in asr_segments:
        speaker = seg["speaker"]
        start, end = seg["start"], seg["end"]
        text = seg["text"]

        dur = end - start

        # Case 1: no active block → start new block
        if current is None:
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "reactions": []
            }
            continue

        # Case 2: same speaker → extend main block
        if speaker == current["speaker"]:
            current["end"] = end
            current["text"] += " " + text
            continue

        # Case 3: interruption by different speaker
        if dur <= reaction_threshold:
            # treat as REACTION
            current["reactions"].append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text
            })
        else:
            # interruption is long → close current block and start new block
            merged.append(current)
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "reactions": []
            }

    # append last block
    if current:
        merged.append(current)

    return merged

# 4. To save final JSON
def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# 5. Main pipeline 
def run_pipeline(asr_json_path, diar_json_path, output_path):
    asr = load_json(asr_json_path)
    diar = load_json(diar_json_path)

    asr_segments = asr["segments"]
    diar_segments = diar["segments"]

    print("Assigning speakers to ASR segments...")
    asr_with_speakers = assign_speakers(asr_segments, diar_segments)

    print("Merging into semantic floor-taking structure...")
    merged_output = merge_floor_taking(asr_with_speakers)

    print("Saving merged output...")
    save_json(merged_output, output_path)

    print("Done! Final JSON saved to:", output_path)