import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import os
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Callable, Optional

from .utils import (
    PerParameterMSE, EarlyStopping,
    save_checkpoint, load_checkpoint, compute_metrics, save_metrics,
    compute_metrics_with_filtering, MEANINGFUL_THRESHOLDS
)


class Trainer:
    """
    Training manager for CNN regression model.
    
    Handles training loop, validation, checkpointing, and metric tracking.
    """
    
    def __init__(self, model, dataloaders, stats, selected_params,
                 learning_rate=1e-3, weight_decay=1e-5, device='cpu',
                 checkpoint_dir='./checkpoints', patience=10,
                 use_amp=True, use_bfloat16=True, compile_model=True,
                 val_every_n_epochs=1, use_tf32=True, accumulation_steps=1,
                 scheduler_type='cosine'):
        """
        Args:
            model: CNNRegressor model
            dataloaders: Dict with 'train', 'val', 'test' DataLoaders
            stats: Statistics dict for normalization
            selected_params: List of parameter names being predicted
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            device: Device to train on ('cpu', 'cuda', 'mps')
            checkpoint_dir: Directory to save checkpoints
            patience: Early stopping patience
            use_amp: Enable automatic mixed precision
            use_bfloat16: Use bfloat16 (faster on M1 than float16)
            compile_model: Use torch.compile() for optimization
            val_every_n_epochs: Validate every N epochs (1 = every epoch)
            use_tf32: Enable TF32 precision for 2x speedup on Ampere/Ada GPUs
            accumulation_steps: Gradient accumulation steps (effective batch = batch * steps)
        """
        self.model = model
        self.dataloaders = dataloaders
        self.stats = stats
        self.selected_params = selected_params
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp and device in ['cuda', 'mps']  # AMP only for GPU
        self.use_bfloat16 = use_bfloat16
        self.accumulation_steps = accumulation_steps
        
        # Enable TF32 precision for ~2x speedup on Ampere/Ada GPUs (A100, L4, etc.)
        if use_tf32 and 'cuda' in str(device):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 precision enabled (2x speedup on Ada/Ampere)")
        
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Compile model for optimization (PyTorch 2.0+) - disabled for MPS (not compatible)
        self.model_compiled = False
        if compile_model and hasattr(torch, 'compile') and device != 'mps':
            try:
                self.model = torch.compile(model, mode='reduce-overhead')
                self.model_compiled = True
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"torch.compile() not available: {e}")
                self.model = model
        else:
            self.model = model
        
        # Store scheduler type
        self.scheduler_type = scheduler_type
        
        # Force validation every epoch for ReduceLROnPlateau (needs to see loss every epoch)
        if scheduler_type == 'plateau' and val_every_n_epochs > 1:
            print(f"ReduceLROnPlateau requires validation every epoch (was {val_every_n_epochs})")
            val_every_n_epochs = 1
        self.val_every_n_epochs = val_every_n_epochs
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == 'plateau':
            # ReduceLROnPlateau - drops LR when validation loss stops improving
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=learning_rate * 0.01
            )
            self.scheduler_steps_per_epoch = False  # Step only on validation
            print("Using ReduceLROnPlateau scheduler (classic)")
        else:
            # Cosine Annealing with Warm Restarts - cycles LR with periodic restarts
            # Better exploration of loss landscape
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,  # First restart at epoch 10
                T_mult=2,  # Double the cycle length after each restart
                eta_min=learning_rate * 0.01  # Minimum LR (1% of initial)
            )
            self.scheduler_steps_per_epoch = True  # Step every epoch
            print("Using CosineAnnealingWarmRestarts scheduler")
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.per_param_metric = PerParameterMSE(stats, selected_params, device=device)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
        
        # AMP gradient scaler (only needed for CUDA, MPS handles this differently)
        self.scaler = GradScaler(device) if self.use_amp and device == 'cuda' else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'per_param_train': {p: [] for p in selected_params},
            'per_param_val': {p: [] for p in selected_params}
        }
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # Move model to device and wrap with DataParallel if multiple GPUs available
        self.model.to(device)
        self.criterion.to(device)

        # Enable DataParallel for multi-GPU training
        if 'cuda' in str(device) and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
    
    def train_epoch(self):
        """Train for one epoch with optional AMP, label smoothing, and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        accumulated_loss = 0.0
        num_batches = 0
        accumulation_counter = 0
        all_per_param_losses = {p: [] for p in self.selected_params}
        max_grad_norm = 0.0  # Track gradient norms for verification

        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {self.current_epoch}', leave=False)

        # Zero gradients at start of epoch
        self.optimizer.zero_grad()

        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Label smoothing: add small noise to targets for regularization
            # This prevents overconfidence and improves generalization
            if self.model.training:
                noise = torch.randn_like(targets) * 0.02  # ε = 0.02
                targets_smooth = torch.clamp(targets + noise, 0.0, 1.0)
            else:
                targets_smooth = targets

            # Use bfloat16 on MPS (faster than float16), float16 on CUDA
            amp_dtype = torch.bfloat16 if self.use_bfloat16 else None

            with autocast(self.device, enabled=self.use_amp, dtype=amp_dtype):
                predictions = self.model(images)
                # Scale loss by accumulation steps for proper gradient accumulation
                loss = self.criterion(predictions, targets_smooth) / self.accumulation_steps

            # Backward pass with AMP - accumulate gradients
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item() * self.accumulation_steps
            accumulation_counter += 1

            # Per-parameter losses (on unscaled loss for accurate tracking)
            with torch.no_grad():
                unscaled_loss_predictions = predictions.detach()
                per_param = self.per_param_metric(unscaled_loss_predictions * self.accumulation_steps, targets)
                for param, val in per_param.items():
                    all_per_param_losses[param].append(val)

            # Gradient step only every accumulation_steps batches
            if accumulation_counter % self.accumulation_steps == 0:
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping with norm tracking
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    max_grad_norm = max(max_grad_norm, grad_norm.item() if torch.isfinite(grad_norm) else 0.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping with norm tracking
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    max_grad_norm = max(max_grad_norm, grad_norm.item() if torch.isfinite(grad_norm) else 0.0)
                    self.optimizer.step()

                # Zero gradients for next accumulation cycle
                self.optimizer.zero_grad()

            # Track metrics
            total_loss += accumulated_loss
            num_batches += 1
            accumulated_loss = 0.0

            pbar.set_postfix({'loss': f'{loss.item() * self.accumulation_steps:.6f}'})

        # Handle any remaining accumulated gradients at end of epoch
        if accumulation_counter % self.accumulation_steps != 0:
            if self.use_amp and self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                max_grad_norm = max(max_grad_norm, grad_norm.item() if torch.isfinite(grad_norm) else 0.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                max_grad_norm = max(max_grad_norm, grad_norm.item() if torch.isfinite(grad_norm) else 0.0)
                self.optimizer.step()

        # Compute average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_per_param = {p: np.mean(v) if v else 0.0 for p, v in all_per_param_losses.items()}

        # Print gradient norm info for verification (every 5 epochs)
        if self.current_epoch % 5 == 0:
            clipped_pct = (max_grad_norm > 1.0) * 100
            print(f"  Grad norm: {max_grad_norm:.3f} {'(CLIPPED)' if max_grad_norm > 1.0 else '(OK)'}  (eff_batch={self.accumulation_steps}x)")

        return avg_loss, avg_per_param
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_per_param_losses = {p: [] for p in self.selected_params}

        with torch.no_grad():
            for images, targets in self.dataloaders['val']:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                amp_dtype = torch.bfloat16 if self.use_bfloat16 else None

                with autocast(self.device, enabled=self.use_amp, dtype=amp_dtype):
                    predictions = self.model(images)
                    loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

                per_param = self.per_param_metric(predictions, targets)
                for param, val in per_param.items():
                    all_per_param_losses[param].append(val)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_per_param = {p: np.mean(v) if v else 0.0 for p, v in all_per_param_losses.items()}

        return avg_loss, avg_per_param
    
    def train(self, epochs: int, progress_callback: Optional[Callable] = None):
        """
        Train the model for specified epochs.

        Args:
            epochs: Number of epochs to train
            progress_callback: Optional callback(epoch, train_loss, val_loss, history)

        Returns:
            dict: Training history
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Parameters: {self.selected_params}")
        print(f"Device: {self.device}")
        if self.use_bfloat16:
            print("Using bfloat16 mixed precision (MPS optimized)")
        print(f"Validating every {self.val_every_n_epochs} epoch(s)")

        last_val_loss = float('inf')
        last_val_per_param = {p: float('inf') for p in self.selected_params}

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch + 1

            # Train
            train_loss, train_per_param = self.train_epoch()

            # Validate only every N epochs (or on last epoch)
            should_validate = (self.current_epoch % self.val_every_n_epochs == 0) or (self.current_epoch == epochs)

            if should_validate:
                val_loss, val_per_param = self.validate()
                last_val_loss = val_loss
                last_val_per_param = val_per_param

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                    save_checkpoint(
                        self.model, self.optimizer, self.current_epoch,
                        train_loss, val_loss, self.stats, self.selected_params,
                        best_path
                    )
                    print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

                # Early stopping check (only when we validate)
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {self.current_epoch}")
                    break
                    
                # Update ReduceLROnPlateau scheduler with validation loss
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
            else:
                # Use last known val_loss and per-param values (repeat them, don't zero them)
                val_loss = last_val_loss
                val_per_param = last_val_per_param
            
            # Update CosineAnnealingWarmRestarts scheduler every epoch
            if self.scheduler_type == 'cosine':
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            for param in self.selected_params:
                self.history['per_param_train'][param].append(train_per_param[param])
                self.history['per_param_val'][param].append(val_per_param[param])

            # Print progress
            if should_validate:
                print(f"Epoch {self.current_epoch}/{epochs} | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}")
            else:
                print(f"Epoch {self.current_epoch}/{epochs} | "
                      f"Train: {train_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"(no validation)")

            # Progress callback for UI updates
            if progress_callback:
                progress_callback(self.current_epoch, train_loss, val_loss, self.history)
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.6f}")
        
        # Save final model
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        save_checkpoint(
            self.model, self.optimizer, self.current_epoch,
            self.history['train_loss'][-1], self.history['val_loss'][-1],
            self.stats, self.selected_params, final_path
        )
        
        # Save history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def evaluate(self, split='test'):
        """
        Evaluate on test or validation set.
        
        Args:
            split: 'test' or 'val'
        
        Returns:
            dict: Evaluation metrics with both full and meaningful versions
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.dataloaders[split]:
                images = images.to(self.device)
                
                predictions = self.model(images)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Build targets dict for filtering (parameter_name -> values array)
        targets_dict = {}
        for i, param in enumerate(self.selected_params):
            targets_dict[param] = targets[:, i]
        
        # Compute comprehensive metrics with filtering for meaningful samples
        metrics = compute_metrics_with_filtering(
            predictions, targets, self.selected_params, targets_dict
        )
        
        # Add raw arrays for plotting
        metrics['predictions'] = predictions.tolist()
        metrics['targets'] = targets.tolist()
        
        # Add targets_dict for filtering info in UI
        metrics['targets_dict'] = {k: v.tolist() for k, v in targets_dict.items()}
        
        return metrics
    
    def predict(self, image):
        """
        Make prediction on a single image.
        
        Args:
            image: Tensor of shape (1, H, W) or (B, 1, H, W)
        
        Returns:
            numpy array: Predicted parameters
        """
        self.model.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(image)
        
        return prediction.cpu().numpy()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint_info = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.device
        )
        self.current_epoch = checkpoint_info['epoch']
        self.best_val_loss = checkpoint_info['val_loss']
        
        # Restore scheduler state to match loaded epoch
        # Only for CosineAnnealingWarmRestarts - ReduceLROnPlateau can't be restored this way
        if self.current_epoch > 0 and self.scheduler_type == 'cosine':
            print(f"Restoring scheduler state to epoch {self.current_epoch}...")
            for _ in range(self.current_epoch):
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  -> Restored LR: {current_lr:.6f}")
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint_info


def create_trainer(model, dataloaders, stats, selected_params,
                   learning_rate=1e-3, weight_decay=1e-5, device='cpu',
                   checkpoint_dir='./checkpoints', patience=10,
                   use_amp=True, use_bfloat16=True, compile_model=True,
                   val_every_n_epochs=1, use_tf32=True, accumulation_steps=1,
                   scheduler_type='cosine'):
    """
    Factory function to create a Trainer instance.

    Args:
        model: CNNRegressor model
        dataloaders: Dict with 'train', 'val', 'test' DataLoaders
        stats: Statistics dict for normalization
        selected_params: List of parameter names
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        patience: Early stopping patience
        use_amp: Enable automatic mixed precision
        use_bfloat16: Use bfloat16 (faster on M1)
        compile_model: Use torch.compile() for optimization
        val_every_n_epochs: Validate every N epochs
        use_tf32: Enable TF32 precision for 2x speedup on Ada/Ampere
        accumulation_steps: Gradient accumulation steps
        scheduler_type: 'cosine' or 'plateau' (ReduceLROnPlateau)

    Returns:
        Trainer instance
    """
    return Trainer(
        model=model,
        dataloaders=dataloaders,
        stats=stats,
        selected_params=selected_params,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        use_amp=use_amp,
        use_bfloat16=use_bfloat16,
        use_tf32=use_tf32,
        accumulation_steps=accumulation_steps,
        compile_model=compile_model,
        val_every_n_epochs=val_every_n_epochs,
        scheduler_type=scheduler_type
    )
