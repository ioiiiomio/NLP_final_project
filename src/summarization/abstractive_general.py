from transformers import BartTokenizer, BartForConditionalGeneration

def load_general_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model


def preprocess_asr(asr_json):
    return " ".join([seg["text"] for seg in asr_json["segments"]])


def generate_general_summary(asr_json, max_length=200):
    tokenizer, model = load_general_model()
    text = preprocess_asr(asr_json)

    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        early_stopping=True,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)