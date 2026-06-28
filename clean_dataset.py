#!/usr/bin/env python3
"""Clean up old dataset before fresh generation."""
import os
import glob
import shutil

dataset_dir = "SynthDataset"

def clean_dataset():
    print("=" * 60)
    print("Dataset Cleanup")
    print("=" * 60)
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory doesn't exist yet. Nothing to clean.")
        return
    
    # Count existing files
    audio_files = glob.glob(os.path.join(dataset_dir, "audio", "*.wav"))
    spec_files = glob.glob(os.path.join(dataset_dir, "spectrograms", "*.png")) + \
                 glob.glob(os.path.join(dataset_dir, "spectrograms", "*.jpg"))
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Found {len(spec_files)} spectrogram files")
    
    # Ask for confirmation (if running interactively)
    response = input("\nDelete all files and start fresh? [y/N]: ").strip().lower()
    
    if response == 'y':
        print("\nDeleting dataset contents...")
        
        # Remove audio files
        for f in audio_files:
            os.remove(f)
        print(f"  Deleted {len(audio_files)} audio files")
        
        # Remove spectrogram files
        for f in spec_files:
            os.remove(f)
        print(f"  Deleted {len(spec_files)} spectrogram files")
        
        # Remove metadata and checkpoint
        for f in ["metadata.csv", "generator_checkpoint.json"]:
            path = os.path.join(dataset_dir, f)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Deleted {f}")
        
        # Clean empty directories
        for subdir in ["audio", "spectrograms"]:
            path = os.path.join(dataset_dir, subdir)
            if os.path.exists(path) and not os.listdir(path):
                os.rmdir(path)
                print(f"  Removed empty {subdir}/ directory")
        
        print("\n✅ Dataset cleaned! Ready for fresh generation.")
        print(f"\nRun: python synthax_batch_generator.py")
    else:
        print("\n❌ Cleanup cancelled.")
        print("To generate anyway (appending), run: python synthax_batch_generator.py")

if __name__ == "__main__":
    clean_dataset()