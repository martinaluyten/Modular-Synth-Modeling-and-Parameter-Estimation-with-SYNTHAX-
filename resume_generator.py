#!/usr/bin/env python3
"""Resume data generation from existing count."""
import time
import os
import gc
import sys
from synthax_synth_randomized import generate_batch

# Configuration - MODIFY THESE
EXISTING_COUNT = 17000      # ← Set to your current file count
FILES_TO_ADD = 50000        # How many MORE files to generate
BATCH_SIZE = 1000
OUTPUT_FOLDER = "SynthDataset"

# Calculate starting point
START_BATCH = EXISTING_COUNT // BATCH_SIZE  # Batch 17 for 17000 files
NUM_BATCHES = FILES_TO_ADD // BATCH_SIZE   # 50 more batches
SEED_OFFSET_MULTIPLIER = BATCH_SIZE

print("=" * 60)
print("SynthAX RESUME Generator")
print("=" * 60)
print(f"Existing files: {EXISTING_COUNT}")
print(f"Files to add: {FILES_TO_ADD}")
print(f"Starting from batch: {START_BATCH}")
print(f"File range: {EXISTING_COUNT + 1} to {EXISTING_COUNT + FILES_TO_ADD}")
print(f"Output folder: {OUTPUT_FOLDER}")
print("=" * 60)

total_generated = 0

for batch_num in range(START_BATCH, START_BATCH + NUM_BATCHES):
    batch_offset = batch_num * BATCH_SIZE
    base_seed = batch_num * SEED_OFFSET_MULTIPLIER
    
    print("\n" + "-" * 60)
    print(f"BATCH {batch_num - START_BATCH + 1}/{NUM_BATCHES} (overall batch {batch_num})")
    print(f"  File range: {batch_offset + 1} to {batch_offset + BATCH_SIZE}")
    print(f"  Base seed: {base_seed}")
    print("-" * 60)
    
    try:
        # ALWAYS append when resuming
        generate_batch(
            batch_offset=batch_offset,
            base_seed=base_seed,
            batch_size=BATCH_SIZE,
            output_folder=OUTPUT_FOLDER,
            append_metadata=True  # ← Always append
        )
        total_generated += BATCH_SIZE
        
        print("  Cleaning up memory...")
        gc.collect()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

print(f"\nAdded {total_generated} new files!")
