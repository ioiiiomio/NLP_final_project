from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache
import torch
import re

# Model Registry for different conversation types
MODEL_REGISTRY = {
    "interview": {
        "model_name": "philschmid/bart-large-cnn-samsum",
        "description": "Best for podcasts, interviews, Q&A",
        "max_input_length": 1024,
        "tokenizer_kwargs": {}
    },
    "social": {
        "model_name": "google/pegasus-cnn_dailymail",
        "description": "Best for casual, emotional, storytelling conversations",
        "max_input_length": 512,
        "tokenizer_kwargs": {"model_max_length": 1024}
    },
    "meeting": {
        "model_name": "mikeadimech/longformer-qmsum-meeting-summarization",
        "description": "Best for formal meetings, decisions, action items",
        "max_input_length": 4096,
        "tokenizer_kwargs": {}
    },
    "general": {
        "model_name": "facebook/bart-large-cnn",
        "description": "Good all-rounder for various text",
        "max_input_length": 1024,
        "tokenizer_kwargs": {}
    }
}

# Cache models to avoid reloading
@lru_cache(maxsize=4)
def load_model(model_name):
    """Cache models to avoid reloading"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def preprocess_asr(asr_json):
    """Extract and concatenate text from segments"""
    return " ".join([seg["text"] for seg in asr_json["segments"]])


def heuristic_detect_conversation_type(text):
    """
    Heuristic fallback detection (always works).
    This is your existing detection logic.
    """
    # Your existing heuristic logic here
    interview_keywords = ['podcast', 'interview', 'welcome to', 'host', 'guest', 'q&a', 'qa']
    social_keywords = ['joke', 'prank', 'felt', 'thought', 'funny', 'awesome', 'story', 'laugh', 'hilarious']
    meeting_keywords = ['meeting', 'agenda', 'action item', 'decision', 'deadline', 'project', 'team', 'discussion']
    
    question_count = text.count('?')
    
    # Check for interview patterns (Q&A heavy or explicit interview terms)
    if any(kw in text for kw in interview_keywords) or question_count > 3:
        return "interview"
    
    # Check for social/emotional content
    if any(kw in text for kw in social_keywords):
        return "social"
    
    # Check for business/meeting content
    if any(kw in text for kw in meeting_keywords):
        return "meeting"
    
    return "general"


def detect_conversation_type(asr_json, use_ml=False):
    """
    Detect conversation type with optional ML enhancement.
    
    Args:
        asr_json: Your transcript data
        use_ml: Whether to use ML detector (if available)
    
    Returns:
        One of: 'interview', 'social', 'meeting', 'general'
    """
    text = preprocess_asr(asr_json).lower()
    
    # Try ML detection if enabled and available
    if use_ml:
        try:
            # Lazy import to avoid dependency if not needed
            from ml_detector import MLConversationDetector
            
            # Initialize detector (create models directory if needed)
            import os
            detector_path = "models/conversation_classifier.joblib"
            
            # Check if model exists
            if os.path.exists(detector_path):
                ml_detector = MLConversationDetector(detector_path)
                
                # Extract segments if available
                segments = asr_json.get('segments', [])
                
                # Use ML prediction
                ml_prediction = ml_detector.predict(text, segments)
                print(f"[INFO] ML detector prediction: {ml_prediction}")
                return ml_prediction
            else:
                print(f"[INFO] ML model not found at {detector_path}, using heuristic")
                
        except ImportError:
            print("[INFO] ml_detector module not available, using heuristic")
        except Exception as e:
            print(f"[WARNING] ML detector failed: {e}, using heuristic")
    
    # Fall back to heuristic detection (ALWAYS AVAILABLE)
    return heuristic_detect_conversation_type(text)


def generate_meeting_summary(asr_json, max_length=256, model_type="auto", force_diarization=True, use_ml_detection=False):
    """
    Generate summary using appropriate model based on conversation type.
    
    Args:
        asr_json: Transcript in your format
        max_length: Maximum summary length
        model_type: 'auto' (detect), 'interview', 'social', 'meeting', 'general'
        force_diarization: Whether to preserve speaker labels in output
        use_ml_detection: Whether to use ML for conversation type detection
    
    Returns:
        Dictionary with summary and metadata
    """
    # Auto-detect if needed
    if model_type == "auto":
        model_type = detect_conversation_type(asr_json, use_ml=use_ml_detection)
    
    # Get model config
    config = MODEL_REGISTRY.get(model_type, MODEL_REGISTRY["general"])
    
    # Load model and tokenizer
    tokenizer, model = load_model(config["model_name"])
    
    # Preprocess text
    raw_text = preprocess_asr(asr_json)
    
    # Special preprocessing based on model type
    if model_type == "interview" and force_diarization:
        # For interviews, keep speaker labels for better context
        text = raw_text  # Already includes speaker labels from your pipeline
    else:
        # For other types, use clean text
        text = raw_text
    
    # Tokenize and generate
    inputs = tokenizer(
        text,
        max_length=config["max_input_length"],
        truncation=True,
        return_tensors="pt",
        **config.get("tokenizer_kwargs", {})
    )
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=50,
            early_stopping=True,
            length_penalty=2.0,
            no_repeat_ngram_size=3
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Post-process summary based on conversation type
    if model_type == "interview" and force_diarization:
        # Clean up but keep structure
        summary = clean_interview_summary(summary)
    elif model_type == "social":
        # Ensure narrative flow
        summary = ensure_narrative_flow(summary)
    elif model_type == "meeting":
        # Ensure meeting summaries are concise
        summary = clean_meeting_summary(summary)
    else:
        # General cleaning
        summary = clean_general_summary(summary)
    
    return {
        "summary": summary,
        "model_used": config["model_name"],
        "model_type": model_type,
        "confidence": "high"  # Could be enhanced with actual confidence scores
    }


def clean_interview_summary(summary):
    """Clean interview summaries while preserving structure"""
    # Remove repetitive phrases
    import re
    
    # Fix common issues
    summary = re.sub(r'\s+', ' ', summary)  # Remove extra spaces
    summary = re.sub(r'(\.\s*){2,}', '. ', summary)  # Remove duplicate periods
    
    # Ensure proper speaker reference format
    summary = re.sub(r'([Ss]peaker\s+\d+):\s*\1:', r'\1:', summary)
    
    # Remove "the speaker says" type phrases
    summary = re.sub(r'(T|t)he (speaker|host|guest) (says|explains|states|mentions) (that )?', '', summary)
    
    return summary.strip()


def ensure_narrative_flow(summary):
    """Ensure social/story summaries have good narrative flow"""
    sentences = summary.split('. ')
    if len(sentences) > 1:
        # Ensure first sentence is complete
        if len(sentences[0].split()) < 4:
            sentences = sentences[1:]
        
        # Join with proper punctuation
        summary = '. '.join(sentences) + ('' if summary.endswith('.') else '.')
    
    return summary


def clean_meeting_summary(summary):
    """Clean meeting summaries"""
    import re
    
    # Remove redundant meeting phrases
    redundant = [
        "in the meeting",
        "during the meeting", 
        "the meeting discussed",
        "the participants discussed"
    ]
    
    for phrase in redundant:
        summary = re.sub(phrase, '', summary, flags=re.IGNORECASE)
    
    # Fix spacing
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # Ensure it starts clean
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    return summary


def clean_general_summary(summary):
    """Clean general summaries"""
    import re
    
    # Remove "the speaker" type phrases
    summary = re.sub(r'^(T|t)he (speaker|person) (says|explains|mentions|states) (that )?', '', summary)
    
    # Fix spacing
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # Ensure proper punctuation
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary


# Backward compatibility function
def generate_meeting_summary_simple(asr_json, max_length=256):
    """Simple wrapper for backward compatibility"""
    result = generate_meeting_summary(asr_json, max_length, model_type="auto", use_ml_detection=False)
    return result["summary"]


# Additional utility functions for your pipeline
def get_conversation_analysis(asr_json):
    """
    Analyze conversation characteristics.
    Useful for your pipeline's metadata.
    """
    segments = asr_json.get('segments', [])
    
    if not segments:
        return {"error": "No segments found"}
    
    # Count speakers (excluding reactions)
    speakers = set()
    question_count = 0
    total_words = 0
    utterance_lengths = []
    
    for seg in segments:
        if seg.get("is_reaction", False):
            continue
        
        speakers.add(seg.get("speaker", "UNKNOWN"))
        
        text = seg.get("text", "")
        if "?" in text:
            question_count += 1
        
        words = len(text.split())
        total_words += words
        utterance_lengths.append(words)
    
    # Detect conversation type
    conv_type = detect_conversation_type(asr_json, use_ml=False)
    
    return {
        "speaker_count": len(speakers),
        "unique_speakers": list(speakers),
        "question_count": question_count,
        "total_words": total_words,
        "avg_utterance_length": sum(utterance_lengths) / len(utterance_lengths) if utterance_lengths else 0,
        "detected_type": conv_type,
        "suggested_model": MODEL_REGISTRY.get(conv_type, MODEL_REGISTRY["general"])["model_name"]
    }


# Example usage at the bottom (for testing)
if __name__ == "__main__":
    # Test with sample data
    sample_json = {
        "segments": [
            {"text": "Welcome to our podcast interview with Dr. Smith."},
            {"text": "Thank you for having me. I'm excited to be here."},
            {"text": "So, what inspired you to become a researcher?"}
        ]
    }
    
    print("Testing conversation type detection...")
    conv_type = detect_conversation_type(sample_json, use_ml=False)
    print(f"Detected type: {conv_type}")
    
    print("\nTesting summary generation...")
    result = generate_meeting_summary(sample_json, max_length=100)
    print(f"Summary: {result['summary']}")
    print(f"Model used: {result['model_used']}")
    print(f"Model type: {result['model_type']}")
    
    print("\nTesting conversation analysis...")
    analysis = get_conversation_analysis(sample_json)
    print(f"Analysis: {analysis}")