import os
import torch

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "SynthDataset")

# Paths
# Note: spec_path in metadata already includes 'spectrograms/' prefix
# So we use dataset_dir as base and join with spec_path directly
PATHS = {
    'metadata': os.path.join(DATASET_DIR, "metadata.csv"),
    'dataset_dir': DATASET_DIR,  # Base dir - join with spec_path from metadata
    'spectrograms': os.path.join(DATASET_DIR, "spectrograms"),  # For UI display
    'checkpoints': os.path.join(os.path.dirname(__file__), "checkpoints"),
    'results': os.path.join(os.path.dirname(__file__), "results")
}

# Create directories if they don't exist
for path_key in ['checkpoints', 'results']:
    os.makedirs(PATHS[path_key], exist_ok=True)

# Hyperparameters - Adjusted for 22GB L4 GPUs
HYPERPARAMS = {
    'learning_rate': 3e-3,
    'batch_size': 256,  # Reduced from 512 to avoid OOM on 22GB GPUs
    'epochs': 500,
    'patience': 15,
    'weight_decay': 1e-5,
    'dropout': 0.3,
    'hidden_dim': 256
}

# Data split ratios (must sum to 1.0)
DATA_SPLIT = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

# Available parameters (must match dataset column suffixes after 'raw_')
# Column names in CSV: raw_midi_f0, raw_cutoff, raw_attack, etc.
ALL_PARAMETERS = [
    'midi_f0', 'cutoff', 'attack', 'decay', 'sustain',
    'release', 'alpha', 'noise', 'sine', 'sqr', 'shape'
]

# Device configuration
def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

DEVICE = get_device()

# Model architecture config
MODEL_CONFIG = {
    'num_params': len(ALL_PARAMETERS),  # Will be overridden based on selected params
    'dropout': 0.5,
    'hidden_dim': 256
}

# Training config - Optimized for 50k dataset on dual L4 GPUs
TRAINING_CONFIG = {
    'random_seed': 42,
    'num_workers': 16,           # Doubled: More CPU workers for faster data loading
    'pin_memory': True,          # Essential for NVIDIA GPU transfer speeds
    'persistent_workers': True,  
    'prefetch_factor': 4,  
    'gradient_clip': 1.0,
    'use_amp': True,  
    'use_bfloat16': True,        # The L4 GPU natively supports bfloat16
    'compile_model': True,  
    'val_every_n_epochs': 3,
    'use_tf32': True,            # TF32 precision for 2x speedup on Ada/Ampere
    'accumulation_steps': 4,     # 2 GPUs: effective batch = 256 * 4 * 2 = 2048
}
