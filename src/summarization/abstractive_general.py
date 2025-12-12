from transformers import BartTokenizer, BartForConditionalGeneration

def generate_general_summary(asr_json, max_length=200):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    text = " ".join([seg["text"] for seg in asr_json["segments"]])
    
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=40,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary.strip()