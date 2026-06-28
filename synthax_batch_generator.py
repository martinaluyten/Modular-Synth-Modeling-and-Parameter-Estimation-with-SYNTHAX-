#!/usr/bin/env python3
"""
SynthAX Batch Generator with Resume Support
Generates synthetic audio files in sequential batches.
Resumes from last completed batch if interrupted.
"""
import time
import os
import gc
import sys
import glob
import json

from synthax_synth_randomized import generate_batch

# =====================================================================
# --- CONFIGURATION ---
# =====================================================================
TOTAL_FILES = 50000      # Total files to generate
BATCH_SIZE = 2000        # Reduced from 5000 to avoid OOM with 22GB GPU
NUM_BATCHES = TOTAL_FILES // BATCH_SIZE

OUTPUT_FOLDER = "SynthDataset"
CHECKPOINT_FILE = os.path.join(OUTPUT_FOLDER, "generator_checkpoint.json")

# Seed calculation: Each batch gets a unique, non-overlapping seed stream
SEED_OFFSET_MULTIPLIER = BATCH_SIZE

# =====================================================================
# --- CHECKPOINT FUNCTIONS ---
# =====================================================================

def load_checkpoint():
    """Load checkpoint to resume from last completed batch."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_completed_batch": -1, "total_files": 0}

def save_checkpoint(batch_num, total_files):
    """Save checkpoint after each batch."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_completed_batch": batch_num, "total_files": total_files}, f)
    print(f"  Checkpoint saved: batch {batch_num + 1} complete")

def get_actual_file_count():
    """Count actual files on disk to verify against checkpoint."""
    audio_files = glob.glob(os.path.join(OUTPUT_FOLDER, "audio", "*.wav"))
    spec_files = glob.glob(os.path.join(OUTPUT_FOLDER, "spectrograms", "*.png"))
    return min(len(audio_files), len(spec_files))

def scan_existing_files():
    """Scan for existing valid files and determine where to resume."""
    audio_files = set(os.path.basename(f).replace('.wav', '') 
                      for f in glob.glob(os.path.join(OUTPUT_FOLDER, "audio", "*.wav")))
    spec_files = set(os.path.basename(f).replace('.png', '') 
                     for f in glob.glob(os.path.join(OUTPUT_FOLDER, "spectrograms", "*.png")))
    
    # Files that have both audio and spectrogram
    valid_files = audio_files & spec_files
    
    if not valid_files:
        return 0, 0
    
    # Find highest consecutive batch
    file_nums = sorted([int(f.split('_')[2]) for f in valid_files])
    
    # Find last complete batch (all files in batch exist)
    last_complete_batch = -1
    for batch_num in range(NUM_BATCHES):
        start_file = batch_num * BATCH_SIZE + 1
        end_file = (batch_num + 1) * BATCH_SIZE
        batch_complete = all(
            f"synth_sound_{i:05d}" in valid_files 
            for i in range(start_file, end_file + 1)
        )
        if batch_complete:
            last_complete_batch = batch_num
        else:
            break
    
    return last_complete_batch, len(valid_files)

# =====================================================================
# --- MAIN BATCH LOOP ---
# =====================================================================

def main():
    print("=" * 60)
    print("SynthAX Batch Generator with Resume")
    print("=" * 60)
    print(f"Total files to generate: {TOTAL_FILES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {NUM_BATCHES}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("=" * 60)

    # Pre-create output directories
    os.makedirs(os.path.join(OUTPUT_FOLDER, "audio"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "spectrograms"), exist_ok=True)
    
    # Check for checkpoint and scan existing files
    checkpoint = load_checkpoint()
    actual_batch, actual_files = scan_existing_files()
    
    # Use the more conservative of checkpoint or actual files
    last_completed = min(checkpoint["last_completed_batch"], actual_batch)
    
    if last_completed >= 0:
        print(f"\n📁 RESUMING from batch {last_completed + 1}")
        print(f"   Checkpoint: batch {checkpoint['last_completed_batch'] + 1}")
        print(f"   Actual files: {actual_files} ({actual_batch + 1} complete batches)")
        print(f"   Will generate batches {last_completed + 2} to {NUM_BATCHES}")
    else:
        print("\n🆕 Starting fresh generation")
        # Clean up any partial files from incomplete batches
        # (files in batch 0 that don't have all 1000 files)
        pass
    
    # Track overall progress
    total_generated = (last_completed + 1) * BATCH_SIZE if last_completed >= 0 else 0
    
    for batch_num in range(last_completed + 1, NUM_BATCHES):
        batch_offset = batch_num * BATCH_SIZE
        base_seed = batch_num * SEED_OFFSET_MULTIPLIER

        print("\n" + "-" * 60)
        print(f"BATCH {batch_num + 1}/{NUM_BATCHES}")
        print(f"  File range: {batch_offset + 1} to {batch_offset + BATCH_SIZE}")
        print(f"  Base seed: {base_seed}")
        print("-" * 60)

        try:
            # Generate this batch
            # Always append metadata (we're resuming, so file exists)
            append_mode = (batch_num > 0) or os.path.exists(os.path.join(OUTPUT_FOLDER, "metadata.csv"))

            generate_batch(
                batch_offset=batch_offset,
                base_seed=base_seed,
                batch_size=BATCH_SIZE,
                output_folder=OUTPUT_FOLDER,
                append_metadata=append_mode
            )

            total_generated += BATCH_SIZE
            
            # Save checkpoint immediately after successful batch
            save_checkpoint(batch_num, total_generated)

            # Force garbage collection between batches to free memory
            print("  Cleaning up memory...")
            gc.collect()

        except Exception as e:
            print(f"  ERROR in batch {batch_num + 1}: {e}")
            print(f"  Checkpoint saved up to batch {batch_num}")
            print(f"  To resume, run this script again.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"COMPLETE! Generated {total_generated} files in {NUM_BATCHES} batches.")
    print(f"Output location: {OUTPUT_FOLDER}/")
    print(f"  - Audio files: {OUTPUT_FOLDER}/audio/")
    print(f"  - Spectrograms: {OUTPUT_FOLDER}/spectrograms/")
    print(f"  - Metadata: {OUTPUT_FOLDER}/metadata.csv")
    print("=" * 60)
    
    # Clean up checkpoint file on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("\n✅ Checkpoint file cleaned up (generation complete)")

    # Verify results
    verify_dataset()


def verify_dataset():
    """Quick verification of generated dataset"""
    import pandas as pd

    metadata_path = os.path.join(OUTPUT_FOLDER, "metadata.csv")

    if not os.path.exists(metadata_path):
        print("WARNING: metadata.csv not found!")
        return

    df = pd.read_csv(metadata_path)
    num_rows = len(df)

    print("\n--- VERIFICATION ---")
    print(f"Metadata rows: {num_rows} (expected: {TOTAL_FILES})")

    if num_rows != TOTAL_FILES:
        print(f"WARNING: Row count mismatch! Expected {TOTAL_FILES}, found {num_rows}")

    # Check parameter ranges
    print("\nParameter distributions (should be uniform 0-1 for normalized params):")
    print(f"  target_f0:       {df['target_f0'].min():.3f} - {df['target_f0'].max():.3f}")
    print(f"  target_cutoff:   {df['target_cutoff'].min():.3f} - {df['target_cutoff'].max():.3f}")
    print(f"  target_attack:   {df['target_attack'].min():.3f} - {df['target_attack'].max():.3f}")
    print(f"  target_decay:    {df['target_decay'].min():.3f} - {df['target_decay'].max():.3f}")
    print(f"  target_sustain:  {df['target_sustain'].min():.3f} - {df['target_sustain'].max():.3f}")
    print(f"  target_release:  {df['target_release'].min():.3f} - {df['target_release'].max():.3f}")
    print(f"  target_alpha:    {df['target_alpha'].min():.3f} - {df['target_alpha'].max():.3f}")
    print(f"  target_noise:    {df['target_noise'].min():.3f} - {df['target_noise'].max():.3f}")
    print(f"  target_sine:     {df['target_sine'].min():.3f} - {df['target_sine'].max():.3f}")
    print(f"  target_sqr:      {df['target_sqr'].min():.3f} - {df['target_sqr'].max():.3f}")
    print(f"  target_shape:    {df['target_shape'].min():.3f} - {df['target_shape'].max():.3f}")

    # Verify file existence (sample check)
    sample_check_size = min(10, num_rows)
    all_exist = True
    for i in range(sample_check_size):
        idx = i * (num_rows // sample_check_size)
        filename = df.iloc[idx]['filename']
        filepath = os.path.join(OUTPUT_FOLDER, "audio", filename)
        if not os.path.exists(filepath):
            print(f"  MISSING: {filename}")
            all_exist = False

    if all_exist:
        print(f"\nFile existence check passed (sampled {sample_check_size} files)")


if __name__ == "__main__":
    main()