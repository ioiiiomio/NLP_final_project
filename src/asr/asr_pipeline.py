#!/usr/bin/env python
# coding: utf-8

# # First we preprocess the audiotracks

# ### This segment was completed by Koshkimbayeva Zhaniya

# In[6]:


import os
import librosa
import soundfile as sf
from tqdm import tqdm

class AudioPreprocessor:
    # load audio and preprocess any format to 16kHZ mono WAV

    def __init__(self, raw_audio_dir="./data/raw_audio", processed_audio_dir="./data/processed_audio"):

        self.raw_audio_dir = raw_audio_dir
        self.processed_audio_dir = processed_audio_dir
        
        # Create directories if they don't exist
        os.makedirs(self.raw_audio_dir, exist_ok=True)
        os.makedirs(self.processed_audio_dir, exist_ok=True)
        
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'] 
        # maybe there are more formats but now we only care abt main ones anyway
    
    def get_audio_files(self):
        """Get all supported audio files from raw_audio directory"""
        audio_files = []
        
        for file in os.listdir(self.raw_audio_dir):
            file_path = os.path.join(self.raw_audio_dir, file)
            
            # Check if it's a file and has supported extension
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in self.supported_formats:
                    audio_files.append({
                        'filename': file,
                        'path': file_path,
                        'extension': ext
                    })
        
        return audio_files
    
    def load_and_resample(self, audio_path, target_sr=16000):

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Resample if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            # Generate output path
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            out_filename = f"{base_name}_16k.wav"
            out_path = os.path.join(self.processed_audio_dir, out_filename)
            
            # Save as WAV
            sf.write(out_path, audio, target_sr)
            
            return out_path
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def batch_preprocess(self):
        """Preprocess all audio files in raw_audio directory"""
        audio_files = self.get_audio_files()
        
        if not audio_files:
            print(f"No audio files found in {self.raw_audio_dir}")
            return []
        
        print(f"Found {len(audio_files)} audio file(s) to process")
        
        processed_files = []
        for audio_info in tqdm(audio_files, desc="Preprocessing audio"):
            processed_path = self.load_and_resample(audio_info['path'])
            if processed_path:
                processed_files.append({
                    'original': audio_info['path'],
                    'processed': processed_path,
                    'filename': audio_info['filename']
                })
        
        return processed_files


# # Then we run Whisper X

# In[7]:


import json
import os
import whisperx
from tqdm import tqdm


class WhisperASR:
    """Handles WhisperX transcription and alignment"""
    
    def __init__(self, model_size="medium.en", output_dir="./data/asr_json", device="cpu"):
        """
        Initialize Whisper ASR
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            output_dir: Directory to save JSON transcripts
            device: Computation device (cpu, cuda)
        """
        self.model_size = model_size
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        print(f"Loading WhisperX model ({model_size})...")
        self.model = whisperx.load_model(
            model_size,
            device=device,
            compute_type="int8",
            vad_method="silero"
        )
    
    def transcribe_file(self, audio_path, align=True):
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            align: Whether to run alignment
            
        Returns:
            ASR result dictionary
        """
        try:
            print(f"\nTranscribing: {os.path.basename(audio_path)}")
            
            # Transcribe
            print("Running transcription...")
            asr_result = self.model.transcribe(audio_path)
            
            # Show progress
            for _ in tqdm(asr_result["segments"], desc="ASR Segments"):
                pass
            
            # Alignment
            if align and asr_result["segments"]:
                print("Running alignment...")
                align_model, metadata = whisperx.load_align_model(
                    asr_result["language"], 
                    device=self.device
                )
                
                aligned = whisperx.align(
                    asr_result["segments"],
                    align_model,
                    metadata,
                    audio_path,
                    device=self.device
                )
                asr_result = aligned
            
            return asr_result
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {str(e)}")
            return None
    
    def save_transcript(self, asr_result, original_filename):
        """Save ASR result to JSON file"""
        if not asr_result:
            return None
        
        # Generate output filename
        base_name = os.path.splitext(original_filename)[0]
        out_filename = f"{base_name}_asr.json"
        out_path = os.path.join(self.output_dir, out_filename)
        
        # Save to JSON
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asr_result, f, indent=2, ensure_ascii=False)
        
        return out_path


# # Utilities for Whisper X

# In[8]:


import json

def print_segments(asr_json, max_segments=10):
    """Print ASR segments in readable format"""
    if 'segments' not in asr_json:
        print("No segments found in ASR result")
        return
    
    segments = asr_json['segments']
    print(f"\nFound {len(segments)} segments:")
    
    for i, seg in enumerate(segments[:max_segments]):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        print(f"[{i+1:3d}] [{start:7.2f}s → {end:7.2f}s] {text}")
    
    if len(segments) > max_segments:
        print(f"... and {len(segments) - max_segments} more segments")

def extract_text(asr_json):
    """Extract full text from ASR result"""
    if 'segments' not in asr_json:
        return ""
    
    return " ".join(seg.get("text", "") for seg in asr_json["segments"])

def to_simple_segments(asr_json):
    """Convert ASR result to simplified format"""
    if 'segments' not in asr_json:
        return []
    
    return [
        {
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "")
        }
        for seg in asr_json["segments"]
    ]

def load_asr_json(json_path):
    """Load ASR JSON file"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {str(e)}")
        return None


# # Define the pipeline for transcription:

# In[9]:


import os
import json
import time
from datetime import datetime

class ASRPipeline:
    """Main ASR pipeline for batch processing"""
    
    def __init__(self, 
                 raw_audio_dir="./data/raw_audio",
                 processed_audio_dir="./data/processed_audio",
                 output_dir="./data/asr_json",
                 model_size="medium.en",
                 device="cpu"):
        """
        Initialize ASR pipeline
        
        Args:
            raw_audio_dir: Directory containing raw audio files
            processed_audio_dir: Directory for processed audio files
            output_dir: Directory for ASR JSON outputs
            model_size: Whisper model size
            device: Computation device
        """
        self.preprocessor = AudioPreprocessor(raw_audio_dir, processed_audio_dir)
        self.whisper_asr = WhisperASR(model_size, output_dir, device)
        self.output_dir = output_dir
        
        # Track processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'processing_time': 0,
            'file_results': []
        }
    
    def process_single_file(self, audio_path):
        """Process a single audio file"""
        start_time = time.time()
        
        # Preprocess audio
        processed_path = self.preprocessor.load_and_resample(audio_path)
        if not processed_path:
            return None
        
        # Transcribe
        asr_result = self.whisper_asr.transcribe_file(processed_path)
        if not asr_result:
            return None
        
        # Save transcript
        original_filename = os.path.basename(audio_path)
        json_path = self.whisper_asr.save_transcript(asr_result, original_filename)
        
        processing_time = time.time() - start_time
        
        return {
            'audio_file': audio_path,
            'processed_audio': processed_path,
            'json_output': json_path,
            'processing_time': processing_time,
            'segments': len(asr_result.get('segments', [])),
            'language': asr_result.get('language', 'unknown')
        }
    
    def batch_process_all(self):
        """Process all audio files in the raw_audio directory"""
        print("\n" + "="*60)
        print("ASR PIPELINE - BATCH PROCESSING")
        print("="*60)
        
        # Get all audio files
        audio_files = self.preprocessor.get_audio_files()
        self.stats['total_files'] = len(audio_files)
        
        if not audio_files:
            print(f"No audio files found in {self.preprocessor.raw_audio_dir}")
            return self.stats
        
        print(f"Found {len(audio_files)} audio file(s) to process")
        
        # Process each file
        overall_start = time.time()
        
        for audio_info in audio_files:
            print(f"\nProcessing: {audio_info['filename']}")
            
            result = self.process_single_file(audio_info['path'])
            
            if result:
                self.stats['processed_files'] += 1
                self.stats['file_results'].append(result)
                print(f"Successfull ⋆𐙚₊˚⊹♡ Saved to: {result['json_output']}")
                print(f"Processing time: {result['processing_time']:.2f}s")
                print(f"Segments: {result['segments']}")
                print(f"Language: {result['language']}")
            else:
                self.stats['failed_files'] += 1
                print(f"Failed to process ˙𐃷˙ {audio_info['filename']}")
        
        # Calculate overall statistics
        self.stats['processing_time'] = time.time() - overall_start
        
        # Print summary
        self.print_summary()
        
        # Save batch report
        self.save_batch_report()
        
        return self.stats
    
    def print_summary(self):
        print("\n" + "="*60)
        print("ASR PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files: {self.stats['total_files']}")
        print(f"Successfully processed: {self.stats['processed_files']}")
        print(f"Failed: {self.stats['failed_files']}")
        print(f"Total processing time: {self.stats['processing_time']:.2f}s")
        
        if self.stats['processed_files'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['processed_files']
            print(f"Average time per file: {avg_time:.2f}s")
    
    def save_batch_report(self): #Saved the Batch Report in JSON(check /asr_json)
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_size': self.whisper_asr.model_size,
            'device': self.whisper_asr.device,
            'statistics': self.stats,
            'processed_files': [
                {
                    'original': os.path.basename(res['audio_file']),
                    'json_output': os.path.basename(res['json_output']) if res.get('json_output') else None,
                    'processing_time': res['processing_time'],
                    'segments': res['segments'],
                    'language': res['language']
                }
                for res in self.stats['file_results']
            ]
        }
        
        report_path = os.path.join(self.output_dir, f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n Batch report saved to: {report_path}")