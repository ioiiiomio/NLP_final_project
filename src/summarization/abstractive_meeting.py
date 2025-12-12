from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "mikeadimech/longformer-qmsum-meeting-summarization"

def load_meeting_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def preprocess_asr(asr_json):
    return " ".join([seg["text"] for seg in asr_json["segments"]])


def generate_meeting_summary(asr_json, max_length=256):
    tokenizer, model = load_meeting_model()
    text = preprocess_asr(asr_json)

    inputs = tokenizer(text, max_length=4096, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        early_stopping=True,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
