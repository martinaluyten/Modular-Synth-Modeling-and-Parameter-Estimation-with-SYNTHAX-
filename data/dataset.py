import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import json


ALL_PARAMETERS = [
    'midi_f0', 'cutoff', 'attack', 'decay', 'sustain',
    'release', 'alpha', 'noise', 'sine', 'sqr', 'shape'
]


class SynthDataset(Dataset):
    def __init__(self, metadata_df, dataset_dir, selected_params=None, 
                 transform=None, compute_stats=False, stats=None):
        """
        Dataset for synthesizer parameter prediction from mel-spectrograms.
        
        Args:
            metadata_df: DataFrame with metadata (filtered for train/val/test)
            dataset_dir: Base dataset directory (contains metadata.csv and spectrograms/)
            selected_params: List of parameter names to predict (subset of ALL_PARAMETERS)
            transform: Torchvision transforms for image preprocessing
            compute_stats: If True, compute and return statistics
            stats: Precomputed statistics dict (mean, std for each raw parameter)
        """
        self.df = metadata_df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.selected_params = selected_params or ALL_PARAMETERS
        self.transform = transform or self._default_transform()
        
        if compute_stats:
            self.stats = self._compute_statistics()
        else:
            self.stats = stats or {}
    
    def _default_transform(self):
        """Default image transforms for mel-spectrograms."""
        return transforms.Compose([
            transforms.Resize((128, 256)),  # Fixed size for CNN
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    
    def _compute_statistics(self):
        """Compute mean and std for normalized target parameters."""
        stats = {}
        for param in self.selected_params:
            # Match the exact same target naming logic
            target_col = f"target_{'f0' if param == 'midi_f0' else param}"
            if target_col in self.df.columns:
                stats[param] = {
                    'mean': float(self.df[target_col].mean()),
                    'std': float(self.df[target_col].std())
                }
        return stats
    
    def get_target_columns(self):
        """Get column names for normalized target parameters."""
        return [f"target_{'f0' if p == 'midi_f0' else p}" for p in self.selected_params]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load spectrogram image
        # spec_path in metadata already includes 'spectrograms/' prefix
        spec_path = os.path.join(self.dataset_dir, row['spec_path'])
        image = Image.open(spec_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Get target values (raw, non-normalized parameters)
        # Get target values (normalized [0, 1] parameters)
        targets = []
        for param in self.selected_params:
            # Catch the 'f0' naming mismatch from the generator
            target_col = f"target_{'f0' if param == 'midi_f0' else param}"
            target_val = row[target_col] 
            targets.append(float(target_val))
        
        targets = np.array(targets, dtype=np.float32)
        
        return image, targets


def create_dataloaders(metadata_csv, dataset_dir, selected_params=None,
                       batch_size=32, num_workers=4,
                       persistent_workers=True, prefetch_factor=2,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       random_seed=42):
    """
    Create train/validation/test dataloaders with statistics computation.

    Args:
        metadata_csv: Path to metadata CSV file
        dataset_dir: Base dataset directory (contains metadata.csv and spectrograms/ subdir)
        selected_params: List of parameters to predict (None = all)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1.0)
        random_seed: Random seed for reproducibility

    Returns:
        dict with 'train', 'val', 'test' dataloaders, 'stats', and 'datasets'
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Load metadata
    df = pd.read_csv(metadata_csv)

    # Filter out any rows with missing files
    # spec_path in metadata already includes 'spectrograms/' prefix
    valid_rows = []
    for idx, row in df.iterrows():
        spec_path = os.path.join(dataset_dir, row['spec_path'])
        if os.path.exists(spec_path):
            valid_rows.append(idx)

    df = df.loc[valid_rows].reset_index(drop=True)
    print(f"Loaded {len(df)} valid samples from metadata")

    # Compute split sizes
    total = len(df)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # Ensure all samples used

    # Create datasets with random split
    np.random.seed(random_seed)
    indices = np.random.permutation(total)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets - compute stats on training set only
    train_dataset = SynthDataset(
        train_df, dataset_dir, selected_params,
        compute_stats=True
    )

    # Use training stats for val and test
    stats = train_dataset.stats

    val_dataset = SynthDataset(
        val_df, dataset_dir, selected_params,
        stats=stats
    )
    test_dataset = SynthDataset(
        test_df, dataset_dir, selected_params,
        stats=stats
    )
    
    # Create dataloaders with M1-optimized settings
    loader_kwargs = {
        'num_workers': num_workers,
        'persistent_workers': persistent_workers if num_workers > 0 else False,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'pin_memory': True if num_workers > 0 else False
    }
    # Remove None values
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        **loader_kwargs
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'stats': stats,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }


def save_stats(stats, filepath):
    """Save statistics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)


def load_stats(filepath):
    """Load statistics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
