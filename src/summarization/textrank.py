import nltk
import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt", quiet=True)


def extract_full_text(asr_json):
    """
    Extracts ALL text from ASR segments and returns one combined document.
    """
    segments = asr_json.get("segments", [])
    texts = [seg["text"].strip() for seg in segments if seg.get("text")]
    return " ".join(texts)


def split_sentences(text):
    """
    Splits the full text into sentences.
    """
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    return sentences


def build_similarity_matrix(sentences):
    """
    TF-IDF based similarity matrix between all sentences.
    """
    if len(sentences) == 0:
        return np.zeros((0, 0))

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)

    sim_matrix = cosine_similarity(tfidf)

    np.fill_diagonal(sim_matrix, 0)

    return sim_matrix


def textrank_summarize(asr_json, num_sentences=4):
    full_text = extract_full_text(asr_json)

    sentences = nltk.sent_tokenize(full_text)

    if len(sentences) <= num_sentences:
        return full_text

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)

    sim_matrix = (tfidf * tfidf.T).toarray()

    scores = sim_matrix.sum(axis=1)

    ranked_idx = scores.argsort()[-num_sentences:][::-1]

    ranked_sentences = [sentences[i] for i in ranked_idx]

    return " ".join(ranked_sentences)
