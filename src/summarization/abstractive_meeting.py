from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache
import torch
import re

MODEL_REGISTRY = {
    "interview": {
        "model_name": "philschmid/bart-large-cnn-samsum",
        "description": "Best for interviews, Q&A, podcasts",
        "max_input_length": 1024,
        "tokenizer_kwargs": {}
    },
    "meeting": {
        "model_name": "mikeadimech/longformer-qmsum-meeting-summarization",
        "description": "Best for formal meetings, decisions",
        "max_input_length": 4096,
        "tokenizer_kwargs": {}
    },
    "social": {
        "model_name": "facebook/bart-large-cnn",  
        "description": "For casual, storytelling conversations",
        "max_input_length": 1024,
        "tokenizer_kwargs": {}
    },
    "general": {
        "model_name": "facebook/bart-large-cnn",
        "description": "Good all-rounder for various text", 
        "max_input_length": 1024,
        "tokenizer_kwargs": {}
    }
}

@lru_cache(maxsize=3) 
def load_model(model_name):
    try:
        print(f"[INFO] Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        print(f"[INFO] Falling back to general model")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        return tokenizer, model

def preprocess_asr(asr_json):
    return " ".join([seg["text"] for seg in asr_json["segments"]])

def heuristic_detect_conversation_type(text):
    text_lower = text.lower()
    
    question_count = text_lower.count('?')
    
    interview_keywords = [
        'interview', 'podcast', 'host', 'guest', 'q&a', 'qa',
        'asked', 'question', 'answer', 'discuss', 'welcome to'
    ]
    
    meeting_keywords = [
        'meeting', 'agenda', 'action item', 'decision',
        'deadline', 'project', 'team', 'discussion'
    ]
    
    social_keywords = [
        'joke', 'funny', 'laugh', 'hilarious', 'story',
        'awesome', 'felt', 'thought', 'cool', 'amazing',
        'prank', 'emotional', 'personal'
    ]
    

    if question_count >= 2 or any(kw in text_lower for kw in interview_keywords):
        return "interview"

    elif any(kw in text_lower for kw in meeting_keywords):
        return "meeting"
    
    elif any(kw in text_lower for kw in social_keywords):
        return "social"

    else:
        return "general"

def detect_conversation_type(asr_json):
    text = preprocess_asr(asr_json).lower()
    return heuristic_detect_conversation_type(text)

def generate_meeting_summary(asr_json, max_length=256, model_type="auto", force_diarization=True):
    if model_type == "auto":
        model_type = detect_conversation_type(asr_json)
    
    print(f"[INFO] Detected type: {model_type}")
    
    if model_type not in MODEL_REGISTRY:
        print(f"[WARNING] '{model_type}' not in registry, using 'general'")
        model_type = "general"
    
    config = MODEL_REGISTRY[model_type]
    print(f"[INFO] Using model: {config['model_name']}")

    tokenizer, model = load_model(config["model_name"])
    
    raw_text = preprocess_asr(asr_json)
    text = raw_text 
    

    inputs = tokenizer(
        text,
        max_length=config["max_input_length"],
        truncation=True,
        return_tensors="pt",
        **config.get("tokenizer_kwargs", {})
    )

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
    
    if model_type == "interview":
        summary = clean_interview_summary(summary)
    elif model_type == "meeting":
        summary = clean_meeting_summary(summary)
    elif model_type == "social":
        summary = clean_social_summary(summary)
    else:
        summary = clean_general_summary(summary)
    
    return {
        "summary": summary,
        "model_used": config["model_name"],
        "model_type": model_type
    }

def clean_interview_summary(summary):
    if not summary:
        return summary
    
    summary = re.sub(r'\s+', ' ', summary)
    summary = re.sub(r'(\.\s*){2,}', '. ', summary)
    robotic_phrases = [
        "the speaker discusses",
        "the conversation covers",
        "in this interview",
        "the interview focuses on",
        "the host and guest discuss"
    ]
    
    for phrase in robotic_phrases:
        if summary.lower().startswith(phrase):
            summary = summary[len(phrase):].strip()
            if summary and summary[0].islower():
                summary = summary[0].upper() + summary[1:]
            break

    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary.strip()

def clean_meeting_summary(summary):
    if not summary:
        return summary

    redundant = [
        "in the meeting",
        "during the meeting", 
        "the meeting discussed",
        "the participants discussed",
        "the meeting was about"
    ]
    
    for phrase in redundant:
        summary = re.sub(phrase, '', summary, flags=re.IGNORECASE)
    summary = re.sub(r'\s+', ' ', summary).strip()

    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    return summary

def clean_social_summary(summary):
    if not summary:
        return summary
    
    summary = re.sub(r'^(T|t)he (person|speaker) (says|explains|mentions|states) (that )?', '', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary

def clean_general_summary(summary):
    if not summary:
        return summary

    summary = re.sub(r'^(T|t)he (speaker|person) (says|explains|mentions|states) (that )?', '', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()

    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary

def generate_meeting_summary_simple(asr_json, max_length=256):
    result = generate_meeting_summary(asr_json, max_length, model_type="auto")
    return result["summary"]

def get_conversation_analysis(asr_json):
    segments = asr_json.get('segments', [])
    
    if not segments:
        return {"error": "No segments found"}
    
    speakers = set()
    question_count = 0
    total_words = 0
    
    for seg in segments:
        if seg.get("is_reaction", False):
            continue
        
        speakers.add(seg.get("speaker", "UNKNOWN"))
        
        text = seg.get("text", "")
        if "?" in text:
            question_count += 1
        
        total_words += len(text.split())
    
    conv_type = detect_conversation_type(asr_json)
    
    return {
        "speaker_count": len(speakers),
        "unique_speakers": list(speakers),
        "question_count": question_count,
        "total_words": total_words,
        "detected_type": conv_type,
        "suggested_model": MODEL_REGISTRY.get(conv_type, MODEL_REGISTRY["general"])["model_name"]
    }