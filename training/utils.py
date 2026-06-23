import torch
import torch.nn as nn
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

# Thresholds for meaningful sample filtering during evaluation
# Decay is irrelevant when sustain > 0.92 (envelope stays flat)
# Shape is irrelevant when sqr < 0.15 (square oscillator too quiet)
MEANINGFUL_THRESHOLDS = {
    'decay': {'condition': 'sustain', 'threshold': 0.9, 'operator': 'le'},
    'shape': {'condition': 'sqr', 'threshold': 0.15, 'operator': 'ge'}
}


class NormalizedMSELoss(nn.Module):
    """
    Normalized Mean Squared Error Loss.
    
    Normalizes each parameter by its standard deviation (computed from training data)
    to balance the contribution of parameters with different scales.
    
    Formula: MSE(pred / std, target / std)
    """
    
    def __init__(self, stats: Dict, selected_params: List[str], device='cpu'):
        """
        Args:
            stats: Dictionary with parameter statistics {param: {'mean': x, 'std': y}}
            selected_params: List of parameter names in order
            device: Device to place std tensor on
        """
        super(NormalizedMSELoss, self).__init__()
        
        # Extract std values for selected parameters
        stds = []
        for param in selected_params:
            std = stats.get(param, {}).get('std', 1.0)
            # Avoid division by zero
            if std < 1e-8:
                std = 1.0
            stds.append(std)
        
        # Register as buffer so it's saved with the model
        self.register_buffer('stds', torch.tensor(stds, dtype=torch.float32, device=device))
    
    def forward(self, predictions, targets):
        """
        Compute normalized MSE loss.
        
        Args:
            predictions: (B, num_params) predicted values
            targets: (B, num_params) ground truth values
        
        Returns:
            loss: Scalar normalized MSE
        """
        # Normalize both predictions and targets by std
        # This balances the scale of different parameters
        pred_normalized = predictions / self.stds
        target_normalized = targets / self.stds
        
        # Compute MSE on normalized values
        loss = torch.mean((pred_normalized - target_normalized) ** 2)
        
        return loss


class PerParameterMSE(nn.Module):
    """
    Compute MSE for each parameter separately for monitoring.
    """
    
    def __init__(self, stats: Dict, selected_params: List[str], device='cpu'):
        super(PerParameterMSE, self).__init__()
        self.selected_params = selected_params
        
        # For unnormalized MSE
        self.register_buffer('stds', torch.ones(len(selected_params), device=device))
    
    def forward(self, predictions, targets):
        """
        Compute per-parameter MSE.
        
        Returns:
            dict: {param_name: mse_value}
        """
        errors = (predictions - targets) ** 2  # (B, num_params)
        mses = torch.mean(errors, dim=0)  # (num_params,)
        
        result = {}
        for i, param in enumerate(self.selected_params):
            result[param] = float(mses[i].cpu())
        
        return result


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                     stats, selected_params, filepath):
    """
    Save training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        train_loss: Training loss value
        val_loss: Validation loss value
        stats: Training statistics dictionary
        selected_params: List of selected parameter names
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'stats': stats,
        'selected_params': selected_params,
        'model_architecture': {
            'num_params': len(selected_params),
            'class': model.__class__.__name__
        }
    }
    
    torch.save(checkpoint, filepath)
    
    # Also save config as JSON for easy inspection
    config_path = filepath.replace('.pt', '_config.json')
    config = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'selected_params': selected_params,
        'stats': stats
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load model on
    
    Returns:
        dict: Checkpoint data (epoch, losses, stats, selected_params)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
        'stats': checkpoint.get('stats', {}),
        'selected_params': checkpoint.get('selected_params', [])
    }


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric (e.g., loss)
        
        Returns:
            bool: True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def compute_metrics(predictions, targets, selected_params):
    """
    Compute comprehensive metrics for evaluation.
    
    Args:
        predictions: numpy array (N, num_params)
        targets: numpy array (N, num_params)
        selected_params: List of parameter names
    
    Returns:
        dict: Metrics dictionary
    """
    metrics = {}
    
    # Overall metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    metrics['overall'] = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae)
    }
    
    # Per-parameter metrics
    for i, param in enumerate(selected_params):
        pred = predictions[:, i]
        targ = targets[:, i]
        
        param_mse = np.mean((pred - targ) ** 2)
        param_rmse = np.sqrt(param_mse)
        param_mae = np.mean(np.abs(pred - targ))
        
        # Relative error (percentage)
        mean_val = np.mean(np.abs(targ))
        if mean_val > 1e-8:
            rel_error = param_mae / mean_val * 100
        else:
            rel_error = 0.0
        
        # R² score
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        if ss_tot > 1e-8:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0
        
        metrics[param] = {
            'MSE': float(param_mse),
            'RMSE': float(param_rmse),
            'MAE': float(param_mae),
            'RelativeError_%': float(rel_error),
            'R2': float(r2)
        }
    
    return metrics


def save_metrics(metrics, filepath):
    """Save metrics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def is_sample_meaningful_for_param(targets_dict: Dict[str, np.ndarray], param_name: str) -> np.ndarray:
    """
    Determine which samples are meaningful for a given parameter based on thresholds.
    
    Args:
        targets_dict: Dictionary mapping parameter names to target value arrays
        param_name: Name of the parameter to check
    
    Returns:
        Boolean array indicating which samples are meaningful for this parameter
    """
    if param_name not in MEANINGFUL_THRESHOLDS:
        # No filtering for this parameter - all samples are meaningful
        n_samples = len(next(iter(targets_dict.values())))
        return np.ones(n_samples, dtype=bool)
    
    config = MEANINGFUL_THRESHOLDS[param_name]
    condition_param = config['condition']
    threshold = config['threshold']
    operator = config['operator']
    
    if condition_param not in targets_dict:
        # Condition parameter not available - all samples meaningful
        n_samples = len(next(iter(targets_dict.values())))
        return np.ones(n_samples, dtype=bool)
    
    condition_values = targets_dict[condition_param]
    
    if operator == 'le':
        # Include samples where condition <= threshold
        return condition_values <= threshold
    elif operator == 'ge':
        # Include samples where condition >= threshold
        return condition_values >= threshold
    else:
        # Unknown operator - all samples meaningful
        n_samples = len(condition_values)
        return np.ones(n_samples, dtype=bool)


def compute_metrics_with_filtering(predictions: np.ndarray, 
                                   targets: np.ndarray, 
                                   selected_params: List[str],
                                   targets_dict: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """
    Compute comprehensive metrics with both 'full' and 'meaningful' versions.
    
    For parameters with conditional relevance (decay, shape), computes:
    - 'full' metrics: all samples (for backward compatibility)
    - 'meaningful' metrics: only samples where the parameter is relevant
    
    Args:
        predictions: numpy array (N, num_params) - predicted values
        targets: numpy array (N, num_params) - ground truth values
        selected_params: List of parameter names in order
        targets_dict: Optional dict mapping param names to target arrays for filtering
    
    Returns:
        dict: Metrics with both 'full' and 'meaningful' versions per parameter
    """
    # Build targets_dict from targets array if not provided
    if targets_dict is None:
        targets_dict = {}
        for i, param in enumerate(selected_params):
            targets_dict[param] = targets[:, i]
    
    metrics = {}
    
    # Overall metrics (always computed on all samples)
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    metrics['overall'] = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae)
    }
    
    # Per-parameter metrics
    for i, param in enumerate(selected_params):
        pred = predictions[:, i]
        targ = targets[:, i]
        
        # Full metrics (all samples)
        param_mse = np.mean((pred - targ) ** 2)
        param_rmse = np.sqrt(param_mse)
        param_mae = np.mean(np.abs(pred - targ))
        
        # Relative error (percentage)
        mean_val = np.mean(np.abs(targ))
        if mean_val > 1e-8:
            rel_error = param_mae / mean_val * 100
        else:
            rel_error = 0.0
        
        # R² score
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        if ss_tot > 1e-8:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0
        
        param_metrics = {
            'MSE': float(param_mse),
            'RMSE': float(param_rmse),
            'MAE': float(param_mae),
            'RelativeError_%': float(rel_error),
            'R2': float(r2)
        }
        
        # Check if this parameter has conditional relevance
        if param in MEANINGFUL_THRESHOLDS:
            # Get meaningful sample mask
            meaningful_mask = is_sample_meaningful_for_param(targets_dict, param)
            n_meaningful = np.sum(meaningful_mask)
            n_total = len(meaningful_mask)
            
            # Store filtering info
            param_metrics['meaningful_samples'] = int(n_meaningful)
            param_metrics['total_samples'] = int(n_total)
            param_metrics['excluded_samples'] = int(n_total - n_meaningful)
            param_metrics['excluded_pct'] = float((n_total - n_meaningful) / n_total * 100) if n_total > 0 else 0.0
            
            # Compute meaningful metrics (only on relevant samples)
            if n_meaningful > 0:
                pred_meaningful = pred[meaningful_mask]
                targ_meaningful = targ[meaningful_mask]
                
                meaningful_mse = np.mean((pred_meaningful - targ_meaningful) ** 2)
                meaningful_rmse = np.sqrt(meaningful_mse)
                meaningful_mae = np.mean(np.abs(pred_meaningful - targ_meaningful))
                
                # Relative error for meaningful samples
                mean_val_meaningful = np.mean(np.abs(targ_meaningful))
                if mean_val_meaningful > 1e-8:
                    meaningful_rel_error = meaningful_mae / mean_val_meaningful * 100
                else:
                    meaningful_rel_error = 0.0
                
                # R² for meaningful samples
                ss_res_meaningful = np.sum((targ_meaningful - pred_meaningful) ** 2)
                ss_tot_meaningful = np.sum((targ_meaningful - np.mean(targ_meaningful)) ** 2)
                if ss_tot_meaningful > 1e-8:
                    meaningful_r2 = 1 - (ss_res_meaningful / ss_tot_meaningful)
                else:
                    meaningful_r2 = 0.0
                
                param_metrics['meaningful'] = {
                    'MSE': float(meaningful_mse),
                    'RMSE': float(meaningful_rmse),
                    'MAE': float(meaningful_mae),
                    'RelativeError_%': float(meaningful_rel_error),
                    'R2': float(meaningful_r2)
                }
            else:
                # No meaningful samples - use NaN
                param_metrics['meaningful'] = {
                    'MSE': float('nan'),
                    'RMSE': float('nan'),
                    'MAE': float('nan'),
                    'RelativeError_%': float('nan'),
                    'R2': float('nan')
                }
        
        metrics[param] = param_metrics
    
    return metrics


def export_metrics_to_csv(metrics: Dict, selected_params: List[str], filepath: str):
    """
    Export metrics to CSV with both full and meaningful versions.
    
    Args:
        metrics: Metrics dict from compute_metrics_with_filtering
        selected_params: List of parameter names
        filepath: Path to save CSV
    """
    rows = []
    
    for param in selected_params:
        if param not in metrics:
            continue
            
        param_metrics = metrics[param]
        
        row = {
            'parameter': param,
            'mse': param_metrics['MSE'],
            'rmse': param_metrics['RMSE'],
            'mae': param_metrics['MAE'],
            'r2': param_metrics['R2'],
            'relative_error_pct': param_metrics['RelativeError_%']
        }
        
        # Add meaningful metrics if available
        if 'meaningful' in param_metrics:
            row['meaningful_mse'] = param_metrics['meaningful']['MSE']
            row['meaningful_rmse'] = param_metrics['meaningful']['RMSE']
            row['meaningful_mae'] = param_metrics['meaningful']['MAE']
            row['meaningful_r2'] = param_metrics['meaningful']['R2']
            row['meaningful_relative_error_pct'] = param_metrics['meaningful']['RelativeError_%']
            row['excluded_samples'] = param_metrics['excluded_samples']
            row['excluded_samples_pct'] = param_metrics['excluded_pct']
        else:
            row['meaningful_mse'] = param_metrics['MSE']
            row['meaningful_rmse'] = param_metrics['RMSE']
            row['meaningful_mae'] = param_metrics['MAE']
            row['meaningful_r2'] = param_metrics['R2']
            row['meaningful_relative_error_pct'] = param_metrics['RelativeError_%']
            row['excluded_samples'] = 0
            row['excluded_samples_pct'] = 0.0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return filepath
