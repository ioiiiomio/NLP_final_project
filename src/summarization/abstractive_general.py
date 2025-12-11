from transformers import BartTokenizer, BartForConditionalGeneration
from functools import lru_cache
import torch
import re

@lru_cache(maxsize=1)
def load_general_model():
    """Load and cache the general model for monologues"""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model


def preprocess_asr(asr_json):
    """Extract and concatenate text from segments"""
    return " ".join([seg["text"] for seg in asr_json["segments"]])


def generate_general_summary(asr_json, max_length=200):
    """
    Generate summary for monologues/single speaker content
    Returns just the summary text for backward compatibility
    """
    tokenizer, model = load_general_model()
    text = preprocess_asr(asr_json)

    # Tokenize
    inputs = tokenizer(
        text, 
        max_length=1024, 
        truncation=True, 
        return_tensors="pt"
    )
    
    # Generate with better parameters
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=40,
        early_stopping=True,
        length_penalty=2.0,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Post-process for better quality
    summary = clean_monologue_summary(summary)
    
    return summary


def generate_general_summary_simple(asr_json, max_length=200):
    """
    Simple wrapper for backward compatibility.
    Returns just the summary text.
    
    Args:
        asr_json: Transcript in your format
        max_length: Maximum summary length
    
    Returns:
        Summary text string
    """
    return generate_general_summary(asr_json, max_length)


def clean_monologue_summary(summary):
    """Clean monologue summaries"""
    # Remove incomplete sentences at the end
    summary = re.sub(r'[^.]$', '.', summary)  # Ensure ends with period
    
    # Fix spacing
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # Remove common artifacts
    artifacts = [
        "the speaker says", "the speaker explains", "according to the speaker",
        "in summary", "to sum up"
    ]
    
    for artifact in artifacts:
        if summary.lower().startswith(artifact):
            summary = summary[len(artifact):].strip().capitalize()
    
    return summary


def analyze_speaker_pattern(segments):
    """
    Analyze if content is truly a monologue or has multiple speakers.
    Used by the main pipeline.
    """
    if not segments:
        return {"is_monologue": True, "confidence": 1.0}
    
    speaker_counts = {}
    total_segments = 0
    
    for seg in segments:
        if seg.get("is_reaction", False):
            continue
        
        speaker = seg.get("speaker", "")
        if speaker:
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            total_segments += 1
    
    if total_segments == 0:
        return {"is_monologue": True, "confidence": 1.0}
    
    # Calculate speaker distribution
    unique_speakers = len(speaker_counts)
    if unique_speakers <= 1:
        return {"is_monologue": True, "confidence": 0.95}
    
    # Check if one speaker dominates
    max_segments = max(speaker_counts.values())
    dominance_ratio = max_segments / total_segments
    
    if dominance_ratio > 0.8:  # One speaker has >80% of segments
        return {"is_monologue": True, "confidence": dominance_ratio}
    else:
        return {"is_monologue": False, "confidence": 1 - dominance_ratio}


def detect_monologue_type(text):
    """
    Detect the type of monologue content.
    Returns: 'story', 'explanation', 'presentation', or 'general'
    """
    text_lower = text.lower()
    
    # Check for story/narrative patterns
    story_keywords = ['story', 'once', 'happened', 'experience', 'memory', 'remember']
    story_indicators = ['i felt', 'i thought', 'i saw', 'i heard', 'then i']
    
    # Check for explanation/instruction patterns
    explanation_keywords = ['how to', 'step', 'guide', 'tutorial', 'explain', 'understand']
    explanation_indicators = ['first', 'second', 'third', 'then', 'next', 'finally']
    
    # Check for presentation patterns
    presentation_keywords = ['presentation', 'speech', 'talk', 'lecture', 'conference']
    formal_words = ['thank you', 'welcome', 'today i will', 'in conclusion']
    
    # Score each type
    scores = {
        'story': 0,
        'explanation': 0,
        'presentation': 0,
        'general': 0
    }
    
    # Story detection
    for word in story_keywords:
        if word in text_lower:
            scores['story'] += 1
    
    for phrase in story_indicators:
        if phrase in text_lower:
            scores['story'] += 2
    
    # Explanation detection
    for word in explanation_keywords:
        if word in text_lower:
            scores['explanation'] += 1
    
    for phrase in explanation_indicators:
        if phrase in text_lower:
            scores['explanation'] += 2
    
    # Presentation detection
    for word in presentation_keywords:
        if word in text_lower:
            scores['presentation'] += 1
    
    for phrase in formal_words:
        if phrase in text_lower:
            scores['presentation'] += 2
    
    # Determine highest score
    max_score = max(scores.values())
    
    if max_score > 0:
        # Return type with highest score
        for content_type, score in scores.items():
            if score == max_score:
                return content_type
    
    return "general"


# Optional: Advanced monologue model for different content types
def generate_specialized_monologue_summary(asr_json, content_type="general", max_length=200):
    """
    Advanced: Use different models based on monologue content type
    content_type: "story", "explanation", "presentation", "general"
    """
    # Map content types to different models or parameters
    configs = {
        "story": {"length_penalty": 1.5, "temperature": 0.7},
        "explanation": {"length_penalty": 2.0, "temperature": 0.3},
        "presentation": {"length_penalty": 1.8, "temperature": 0.5},
        "general": {"length_penalty": 2.0, "temperature": 0.7}
    }
    
    config = configs.get(content_type, configs["general"])
    
    tokenizer, model = load_general_model()
    text = preprocess_asr(asr_json)
    
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=40,
        early_stopping=True,
        length_penalty=config["length_penalty"],
        no_repeat_ngram_size=3,
        temperature=config["temperature"]
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return clean_monologue_summary(summary)


if __name__ == "__main__":
    # Test the functions
    test_json = {
        "segments": [
            {"text": "Hello, this is a test monologue."},
            {"text": "I want to tell you about my experience."},
            {"text": "It was really amazing and I learned a lot."}
        ]
    }
    
    print("Testing generate_general_summary...")
    summary = generate_general_summary(test_json)
    print(f"Summary: {summary}")
    
    print("\nTesting analyze_speaker_pattern...")
    test_segments = [
        {"speaker": "SPEAKER_01", "text": "Hello", "is_reaction": False},
        {"speaker": "SPEAKER_01", "text": "How are you?", "is_reaction": False},
    ]
    analysis = analyze_speaker_pattern(test_segments)
    print(f"Analysis: {analysis}")
    
    print("\nTesting detect_monologue_type...")
    text = "I want to tell you a story about my childhood."
    mono_type = detect_monologue_type(text)
    print(f"Type: {mono_type}")