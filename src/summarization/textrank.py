import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download("punkt", quiet=True)

def clean_sentence_for_analysis(sentence):
    filler_words = [
        r'\blike\b', r'\byou know\b', r'\bum\b', r'\buh\b',
        r'\bjust\b', r'\bactually\b', r'\bi mean\b', r'\bi guess\b',
        r'\bkind of\b', r'\bsort of\b'
    ]
    
    cleaned = sentence.lower()
    for filler in filler_words:
        cleaned = re.sub(filler, '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def extract_full_text(asr_json):
    segments = asr_json.get("segments", [])
    texts = [seg["text"].strip() for seg in segments if seg.get("text")]
    return " ".join(texts)

def textrank_summarize(asr_json, num_sentences=4):
    full_text = extract_full_text(asr_json)

    original_sentences = nltk.sent_tokenize(full_text)
    
    if len(original_sentences) <= num_sentences:
        return full_text
    
    cleaned_sentences = [clean_sentence_for_analysis(s) for s in original_sentences]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(cleaned_sentences)
    
    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0)
    
    scores = sim_matrix.sum(axis=1)

    ranked_idx = scores.argsort()[-num_sentences:][::-1]

    selected_indices = sorted(ranked_idx)
    ranked_sentences = [original_sentences[i] for i in selected_indices]
    
    return " ".join(ranked_sentences)