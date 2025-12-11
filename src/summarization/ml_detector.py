"""
Optional ML-based conversation type detector.
Add this later for improved accuracy.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class MLConversationDetector:
    def __init__(self, model_path=None):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        if model_path and os.path.exists(model_path):
            self.classifier = joblib.load(model_path)
            self.is_trained = True
        else:
            self.classifier = RandomForestClassifier(n_estimators=100)
            self.is_trained = False
        
        # Conversation type labels
        self.labels = ["interview", "social", "meeting", "general"]
        
        # Example features for training
        self.feature_names = [
            "question_count", "speaker_count", "avg_sentence_length",
            "emotional_words", "formal_words", "turn_changes"
        ]
    
    def extract_features(self, transcript_text, segments=None):
        """Extract features from transcript"""
        features = []
        
        # 1. Question count
        question_count = transcript_text.count('?')
        features.append(question_count)
        
        # 2. Speaker count (if segments provided)
        if segments:
            speakers = set(seg["speaker"] for seg in segments if not seg.get("is_reaction", False))
            speaker_count = len(speakers)
        else:
            speaker_count = 1
        features.append(speaker_count)
        
        # 3. Average sentence length
        sentences = transcript_text.split('.')
        avg_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        features.append(avg_length)
        
        # 4. Emotional words count
        emotional_words = ["felt", "thought", "happy", "sad", "excited", "angry", "funny", "joke"]
        emotional_count = sum(transcript_text.lower().count(word) for word in emotional_words)
        features.append(emotional_count)
        
        # 5. Formal words count
        formal_words = ["meeting", "agenda", "action", "decision", "deadline", "project"]
        formal_count = sum(transcript_text.lower().count(word) for word in formal_words)
        features.append(formal_count)
        
        # 6. Turn changes (approximate)
        if segments:
            turns = 0
            prev_speaker = None
            for seg in segments:
                if seg.get("is_reaction", False):
                    continue
                if prev_speaker and seg["speaker"] != prev_speaker:
                    turns += 1
                prev_speaker = seg["speaker"]
            features.append(turns)
        else:
            features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, transcript_text, segments=None):
        """Predict conversation type"""
        if not self.is_trained:
            # Fall back to heuristic
            return self.heuristic_predict(transcript_text, segments)
        
        features = self.extract_features(transcript_text, segments)
        prediction_idx = self.classifier.predict(features)[0]
        
        return self.labels[prediction_idx]
    
    def heuristic_predict(self, transcript_text, segments=None):
        """Heuristic fallback"""
        text_lower = transcript_text.lower()
        
        # Check for interview patterns
        if "podcast" in text_lower or "interview" in text_lower or transcript_text.count('?') > 3:
            return "interview"
        
        # Check for social patterns
        social_words = ["joke", "prank", "felt", "thought", "funny", "story"]
        if any(word in text_lower for word in social_words):
            return "social"
        
        # Check for meeting patterns
        meeting_words = ["meeting", "agenda", "action item", "decision"]
        if any(word in text_lower for word in meeting_words):
            return "meeting"
        
        return "general"
    
    def train(self, X, y):
        """Train the classifier"""
        self.classifier.fit(X, y)
        self.is_trained = True
    
    def save(self, path):
        """Save trained model"""
        joblib.dump(self.classifier, path)