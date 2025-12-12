import os
import sys
import json
import argparse
from pathlib import Path


from asr_pipeline import ASRPipeline, extract_text, load_asr_json, print_segments

def main():
    parser = argparse.ArgumentParser(
        description="ASR Pipeline for batch audio processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--raw_audio_dir", 
                       default="./data/raw_audio",
                       help="Directory containing raw audio files (default: ./data/raw_audio)")
    
    parser.add_argument("--processed_audio_dir", 
                       default="./data/processed_audio",
                       help="Directory for processed audio files (default: ./data/processed_audio)")
    
    parser.add_argument("--output_dir", 
                       default="./data/asr_json",
                       help="Directory for ASR JSON outputs (default: ./data/asr_json)")
    
    parser.add_argument("--model_size", 
                       default="medium.en",
                       choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                                "medium", "medium.en", "large", "large-v1", "large-v2"],
                       help="Whisper model size (default: medium.en)")
    
    parser.add_argument("--device", 
                       default="cpu",
                       choices=["cpu", "cuda", "gpu"],
                       help="Computation device (default: cpu)")
    
    parser.add_argument("--single_file", 
                       type=str,
                       help="Process a single file instead of batch processing")
    
    parser.add_argument("--show_transcript",
                       action="store_true",
                       help="Show transcript after processing (for single file mode)")
    
    parser.add_argument("--quiet",
                       action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print("=" * 60)
        print("ASR PIPELINE - Automatic Speech Recognition")
        print("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = ASRPipeline(
            raw_audio_dir=args.raw_audio_dir,
            processed_audio_dir=args.processed_audio_dir,
            output_dir=args.output_dir,
            model_size=args.model_size,
            device=args.device
        )
    except Exception as e:
        print(f"˙𐃷˙ Error initializing pipeline: {e}")
        return 1
    
    # Process single file or batch
    if args.single_file:
        if not args.quiet:
            print(f"⋆.𐙚 ̊   Processing single file: {args.single_file}")
        
        # Check if file exists
        if not os.path.exists(args.single_file):
            print(f"˙𐃷˙ Error: File not found: {args.single_file}")
            return 1
        
        # Process the file
        result = pipeline.process_single_file(args.single_file)
        
        if result and result.get('json_output'):
            if not args.quiet:
                print(f"(˶˃ ᵕ ˂˶)  Successfully processed: {os.path.basename(args.single_file)}")
                print(f" Output: {result['json_output']}")
                print(f" Processing time: {result['processing_time']:.2f}s")
                print(f" Segments: {result['segments']}")
                print(f" Language: {result.get('language', 'unknown')}")
            
            # Load and optionally display the result
            asr_json = load_asr_json(result['json_output'])
            if asr_json:
                # Always save text transcript
                text = extract_text(asr_json)
                txt_path = result['json_output'].replace('.json', '.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                if not args.quiet:
                    print(f"\n Text transcript saved to: {txt_path}")
                
                # Show transcript if requested
                if args.show_transcript:
                    print("\n" + "=" * 60)
                    print("TRANSCRIPT")
                    print("=" * 60)
                    print_segments(asr_json, max_segments=20)
        else:
            print(f"˙𐃷˙ Failed to process {args.single_file}")
            return 1
            
    else:
        # Batch process all files
        if not args.quiet:
            print(f"(๑>؂•̀๑) Processing all audio files in: {args.raw_audio_dir}")
            print("-" * 40)
        
        stats = pipeline.batch_process_all()
        
        if not args.quiet and stats.get('processed_files', 0) > 0:
            print("\n" + "=" * 60)
            print("BATCH PROCESSING COMPLETE")
            print("=" * 60)
            print(f"(˶˃ ᵕ ˂˶)  Successfully processed: {stats['processed_files']} file(s)")
            print(f"˙𐃷˙ Failed: {stats['failed_files']} file(s)")
            print(f" Total time: {stats['processing_time']:.2f}s")
            
            if stats['processed_files'] > 0:
                avg_time = stats['processing_time'] / stats['processed_files']
                print(f" Average per file: {avg_time:.2f}s")
            
            print(f"\n Results saved in: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)