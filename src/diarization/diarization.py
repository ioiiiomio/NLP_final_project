from pyannote.audio import Pipeline
import json, os

HF_TOKEN = "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"
   
def diarize(audio_path): 
   pipeline = Pipeline.from_pretrained(
   "pyannote/speaker-diarization-3.1",
   use_auth_token="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")
   diarization = pipeline(audio_path)
   return diarization

def diarization_json(audio_path): 
   segments = []
   diarization = diarize(audio_path) 
   for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })

   audio_filename = os.path.basename(audio_path)       # "meeting01.wav"
   base_name = os.path.splitext(audio_filename)[0]     # "meeting01"
   output_name = f"{base_name}_diarization.json"       # "meeting01_diarization.json"
    
   output_path = os.path.join("./data/diarizations", output_name)

    # Ensure output folder exists
   os.makedirs("./data/diarizations", exist_ok=True)
   
   # Save JSON
   with open(output_path, "w") as f:
        json.dump({"segments": segments}, f, indent=2)

   print("Diarization JSON saved to:", output_path)
   return output_path
