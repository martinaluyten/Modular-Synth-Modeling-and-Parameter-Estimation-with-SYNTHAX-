import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import json
from PIL import Image
import io

os.environ["CUDA_VISIBLE_DEVICES"] = "0,6"  # Use 2 free L4 GPUs for 2x speedup

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders, ALL_PARAMETERS
from models.cnn_regressor import create_model
from training.train import create_trainer
from training.utils import compute_metrics_with_filtering, export_metrics_to_csv, MEANINGFUL_THRESHOLDS
from inference_utils import (
    decode_normalized_params,
    synthesize_waveform,
    waveform_to_wav_bytes,
    waveform_to_spectrogram_image,
    prepare_image_tensor,
    extract_raw_params_from_row,
    params_dict_to_dataframe,
    predict_and_synthesize,
    synthesize_from_raw_params,
    predict_from_image,
    generate_and_predict,
    spectrogram_diff_image,
    waveform_diff_wav_bytes
)
from config import PATHS, HYPERPARAMS, DEVICE, TRAINING_CONFIG

# Page configuration
st.set_page_config(
    page_title="SynthAX CNN Parameter Predictor",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'dataloaders' not in st.session_state:
    st.session_state.dataloaders = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'model' not in st.session_state:
    st.session_state.model = None


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# Parameter display names (for prettier UI)
PARAM_DISPLAY_NAMES = {
    'midi_f0': 'f0 (MIDI note)',
    'cutoff': 'Cutoff (Hz)',
    'attack': 'Attack',
    'decay': 'Decay',
    'sustain': 'Sustain',
    'release': 'Release',
    'alpha': 'Alpha',
    'noise': 'Noise',
    'sine': 'Sine',
    'sqr': 'Square',
    'shape': 'Shape'
}

def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.title("🎹 Training Controls")
        
        st.markdown("---")
        
        # Parameter Selection
        st.subheader("Parameter Selection")
        st.markdown("Select which parameters to predict:")
        
        selected_params = []
        cols = st.columns(2)
        for i, param in enumerate(ALL_PARAMETERS):
            with cols[i % 2]:
                display_name = PARAM_DISPLAY_NAMES.get(param, param.capitalize())
                if st.checkbox(display_name, value=True, key=f"param_{param}"):
                    selected_params.append(param)
        
        st.markdown("---")
        
        # Hyperparameters
        st.subheader("Hyperparameters")
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2],
            value=HYPERPARAMS['learning_rate'],
            format_func=lambda x: f"{x:g}"
        )
        
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128, 256, 512],
            value=HYPERPARAMS['batch_size']
        )
        
        epochs = st.slider("Epochs", min_value=10, max_value=500, value=HYPERPARAMS['epochs'])
        
        patience = st.slider("Early Stopping Patience", min_value=5, max_value=50, value=HYPERPARAMS['patience'])
        
        weight_decay = st.select_slider(
            "Weight Decay",
            options=[0, 1e-6, 1e-5, 1e-4, 1e-3],
            value=HYPERPARAMS['weight_decay'],
            format_func=lambda x: f"{x:.0e}" if x > 0 else "0"
        )
        
        scheduler_choice = st.selectbox(
            "LR Scheduler",
            ["Cosine Annealing (Warm Restarts)", "Reduce on Plateau (Classic)"],
            help="Cosine: cycles LR down then restarts. Plateau: drops LR when loss plateaus."
        )
        
        st.markdown("---")
        
        # Data split
        st.subheader("Data Split")
        train_ratio = st.slider("Train %", 50, 80, 70) / 100
        val_ratio = st.slider("Validation %", 10, 25, 15) / 100
        test_ratio = 1.0 - train_ratio - val_ratio
        st.write(f"Test: {test_ratio*100:.0f}%")
        
        if test_ratio < 0.05:
            st.error("Invalid split! Adjust train/val percentages.")
        
        st.markdown("---")
        
        # Device info
        device = get_device()
        st.info(f"Device: **{device.upper()}**")
        
        return {
            'selected_params': selected_params,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'patience': patience,
            'weight_decay': weight_decay,
            'scheduler_choice': scheduler_choice,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'device': device
        }

def render_data_exploration():
    """Render data exploration tab."""
    st.markdown("<div class='section-header'>📊 Dataset Exploration</div>", unsafe_allow_html=True)
    
    # Load sample data
    try:
        df = pd.read_csv(PATHS['metadata'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"**Total samples:** {len(df)}")
            
            # Show parameter distributions
            raw_cols = [c for c in df.columns if c.startswith('raw_')]
            selected_col = st.selectbox(
                "Select parameter to visualize",
                [c.replace('raw_', '') for c in raw_cols]
            )
            
            fig = px.histogram(
                df, x=f'raw_{selected_col}',
                nbins=50,
                title=f"Distribution of {selected_col}",
                labels={f'raw_{selected_col}': f'{selected_col} (raw value)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sample Spectrogram")
            
            # Random sample selector
            sample_idx = st.number_input("Sample index", 0, len(df)-1, 0)
            row = df.iloc[sample_idx]
            
            spec_path = os.path.join(PATHS['spectrograms'], row['spec_path'])
            if os.path.exists(spec_path):
                img = Image.open(spec_path)
                st.image(img, caption=row['filename'], use_container_width=True)
            else:
                st.warning("Spectrogram file not found")
            
            # Show parameters
            st.markdown("**Parameters:**")
            for col in raw_cols[:6]:  # Show first 6
                param_name = col.replace('raw_', '')
                st.write(f"• {param_name}: {row[col]:.4f}")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")


def plot_training_charts(history, skip_first=3):
    """
    Plot training charts, optionally skipping first N epochs.
    
    Args:
        history: Training history dict
        skip_first: Number of initial epochs to skip (to avoid skewing graphs)
    
    Returns:
        plotly Figure
    """
    if 'train_loss' not in history or len(history['train_loss']) == 0:
        return None
    
    epochs_completed = len(history['train_loss'])
    start_idx = min(skip_first, epochs_completed - 1)
    
    # Create x values starting from skip_first+1
    x = list(range(start_idx + 1, epochs_completed + 1))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Loss Curves (from epoch {start_idx + 1})", "Learning Rate"),
        column_widths=[0.6, 0.4]
    )
    
    # Slice data to skip first N epochs
    train_slice = history['train_loss'][start_idx:]
    val_slice = history['val_loss'][start_idx:]
    
    fig.add_trace(
        go.Scatter(x=x, y=train_slice, name='Train Loss', mode='lines', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=val_slice, name='Val Loss', mode='lines', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    if 'learning_rate' in history and len(history['learning_rate']) > start_idx:
        lr_slice = history['learning_rate'][start_idx:]
        fig.add_trace(
            go.Scatter(x=x, y=lr_slice, name='LR', mode='lines', line=dict(color='green')),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=True)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
    
    return fig


def plot_parameter_scatter(history, selected_params, skip_first=3):
    """
    Plot per-parameter validation MSE scatter plot.
    
    Args:
        history: Training history dict
        selected_params: List of parameter names
        skip_first: Number of initial epochs to skip
    
    Returns:
        plotly Figure
    """
    if 'per_param_val' not in history:
        return None
    
    epochs_completed = len(history['train_loss'])
    start_idx = min(skip_first, epochs_completed - 1)
    x = list(range(start_idx + 1, epochs_completed + 1))
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for i, param in enumerate(selected_params):
        if param in history['per_param_val'] and len(history['per_param_val'][param]) > start_idx:
            val_slice = history['per_param_val'][param][start_idx:]
            fig.add_trace(go.Scatter(
                x=x, y=val_slice,
                name=param,
                mode='lines',
                line=dict(color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        title=f"Per-Parameter Validation MSE (from epoch {start_idx + 1})",
        xaxis_title="Epoch",
        yaxis_title="Validation MSE",
        height=400,
        showlegend=True
    )
    
    return fig


def render_training_control(config):
    """Render training control tab."""
    st.markdown("<div class='section-header'>🚀 Training Control</div>", unsafe_allow_html=True)
    
    # Check if parameters selected
    if len(config['selected_params']) == 0:
        st.warning("⚠️ Please select at least one parameter in the sidebar!")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Selected Parameters", len(config['selected_params']))
    with col2:
        st.metric("Learning Rate", f"{config['learning_rate']:.0e}")
    with col3:
        st.metric("Epochs", config['epochs'])
    
    st.markdown("---")
    
    # Training buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    
    with btn_col1:
        start_training = st.button("▶️ Start Training", type="primary", disabled=st.session_state.training_in_progress)
    
    with btn_col2:
        resume_mode = st.radio(
            "Resume from:",
            ["Best Model", "Final Model"],
            horizontal=True,
            key="resume_mode",
            disabled=st.session_state.training_in_progress
        )
        resume_training = st.button("⏭️ Resume Training", type="primary", disabled=st.session_state.training_in_progress)

    with btn_col3:
        if st.button("⏹️ Stop Training"):
            st.session_state.training_in_progress = False
            st.rerun()
    
    with btn_col4:
        load_checkpoint = st.button("📂 Load Best Model")
    
    # Progress containers
    progress_container = st.container()
    live_charts_container = st.container()
    param_table_container = st.container()
    
    if start_training or resume_training:
        st.session_state.training_in_progress = True
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_text = st.empty()
        
        # Create placeholders for live charts
        with live_charts_container:
            st.markdown("### 📊 Live Training Progress")
            loss_chart_placeholder = st.empty()
            param_chart_placeholder = st.empty()
        
        # Create placeholder for live parameter table
        with param_table_container:
            st.markdown("### 📋 Live Parameter Validation MSE")
            param_table_placeholder = st.empty()
        
        # Create dataloaders with M1-optimized settings
        status_text.text("Loading data...")
        dataloaders = create_dataloaders(
            metadata_csv=PATHS['metadata'],
            dataset_dir=PATHS['dataset_dir'],
            selected_params=config['selected_params'],
            batch_size=config['batch_size'],
            num_workers=TRAINING_CONFIG['num_workers'],
            persistent_workers=TRAINING_CONFIG.get('persistent_workers', True),
            prefetch_factor=TRAINING_CONFIG.get('prefetch_factor', 2),
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio']
        )
        
        st.session_state.dataloaders = dataloaders
        
        # Create model
        status_text.text("Creating model...")
        model = create_model(
            num_params=len(config['selected_params']),
            dropout=HYPERPARAMS['dropout'],
            hidden_dim=HYPERPARAMS['hidden_dim'],
            device=config['device']
        )
        st.session_state.model = model
        
        # Create trainer with M1-optimized settings
        status_text.text("Initializing trainer...")
        scheduler_type = 'cosine' if 'Cosine' in config['scheduler_choice'] else 'plateau'
        
        # Create trainer with M1-optimized settings
        status_text.text("Initializing trainer...")
        trainer = create_trainer(
            model=model,
            dataloaders=dataloaders,
            stats=dataloaders['stats'],
            selected_params=config['selected_params'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=config['device'],
            checkpoint_dir=PATHS['checkpoints'],
            patience=config['patience'],
            use_amp=TRAINING_CONFIG.get('use_amp', True),
            use_bfloat16=TRAINING_CONFIG.get('use_bfloat16', True),
            compile_model=TRAINING_CONFIG.get('compile_model', True),
            val_every_n_epochs=TRAINING_CONFIG.get('val_every_n_epochs', 2),
            use_tf32=TRAINING_CONFIG.get('use_tf32', True),
            accumulation_steps=TRAINING_CONFIG.get('accumulation_steps', 1),
            scheduler_type=scheduler_type
        )
        st.session_state.trainer = trainer

        if resume_training:
            status_text.text("Restoring weights and optimizer state...")
            resume_model = st.session_state.get('resume_mode', 'Best Model')
            checkpoint_file = 'best_model.pt' if resume_model == 'Best Model' else 'final_model.pt'
            checkpoint_path = os.path.join(PATHS['checkpoints'], checkpoint_file)
            if os.path.exists(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)
                st.info(f"Resumed from {resume_model.lower().replace('_', ' ')}")
            else:
                st.error(f"No {checkpoint_file} found to resume from!")
                st.session_state.training_in_progress = False
                st.rerun()
        
        # Training progress callback - updates live charts
        def progress_callback(epoch, train_loss, val_loss, history):
            progress = epoch / config['epochs']
            progress_bar.progress(min(progress, 1.0))
            epoch_text.text(f"Epoch {epoch}/{config['epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            st.session_state.history = history
            
            # Only update charts after we have enough data (skip_first + 1 epochs)
            if epoch >= 4 and len(history['train_loss']) >= 4:
                # Update loss charts (skipping first 3 epochs)
                loss_fig = plot_training_charts(history, skip_first=3)
                if loss_fig:
                    loss_chart_placeholder.plotly_chart(loss_fig, use_container_width=True, key=f"live_loss_{epoch}")
                
                # Update parameter validation MSE scatter
                param_fig = plot_parameter_scatter(history, config['selected_params'], skip_first=3)
                if param_fig:
                    param_chart_placeholder.plotly_chart(param_fig, use_container_width=True, key=f"live_param_{epoch}")
                
                # Update parameter table with latest validation MSE values
                if 'per_param_val' in history and len(history['per_param_val'][config['selected_params'][0]]) > 0:
                    param_data = []
                    for param in config['selected_params']:
                        if param in history['per_param_val']:
                            val_values = history['per_param_val'][param]
                            if len(val_values) > 0:
                                latest_mse = val_values[-1]
                                # Calculate improvement from 3 epochs ago (if available)
                                improvement = "N/A"
                                if len(val_values) >= 4:
                                    prev_mse = val_values[-4] if len(val_values) >= 4 else val_values[0]
                                    if prev_mse > 0:
                                        pct_change = ((latest_mse - prev_mse) / prev_mse) * 100
                                        improvement = f"{pct_change:+.1f}%"
                                
                                param_data.append({
                                    'Parameter': param,
                                    'Latest Val MSE': f"{latest_mse:.6f}",
                                    'Trend (vs 3 epochs ago)': improvement
                                })
                    
                    if param_data:
                        param_df = pd.DataFrame(param_data)
                        param_table_placeholder.dataframe(param_df, use_container_width=True, hide_index=True)
        
        # Train
        status_text.text("Training in progress...")
        history = trainer.train(epochs=config['epochs'], progress_callback=progress_callback)
        
        st.session_state.training_in_progress = False
        st.session_state.history = history
        status_text.text("Training complete!")
        st.success("✅ Training finished successfully!")
        
        # Clear the live placeholders and show final charts
        loss_chart_placeholder.empty()
        param_chart_placeholder.empty()
        param_table_placeholder.empty()
        
        # Auto-evaluate on test set
        st.info("Evaluating on test set...")
        eval_results = trainer.evaluate('test')
        st.session_state.evaluation_results = eval_results
        
        # Auto-export metrics to CSV
        try:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%dT%H-%M-%S')
            export_path = os.path.join(results_dir, f'{timestamp}_metrics.csv')
            export_metrics_to_csv(eval_results, config['selected_params'], export_path)
            st.success(f"📁 Metrics auto-exported to: {export_path}")
        except Exception as e:
            st.warning(f"Could not auto-export metrics: {e}")
    
    # Display final training charts with adjustable start epoch
    if st.session_state.history and not st.session_state.training_in_progress:
        with live_charts_container:
            st.markdown("### Training Progress")
            history = st.session_state.history

            if 'train_loss' in history and len(history['train_loss']) > 0:
                # Allow user to select starting epoch for better visualization
                max_epoch = len(history['train_loss'])
                default_skip = min(3, max_epoch - 1)

                col1, col2 = st.columns([1, 3])
                with col1:
                    skip_first = st.number_input(
                        "Start from epoch",
                        min_value=0,
                        max_value=max_epoch - 1,
                        value=default_skip,
                        step=1,
                        help="Skip initial epochs to better see convergence"
                    )
                with col2:
                    if skip_first > 0:
                        st.caption(f"📊 Showing epochs {skip_first + 1} to {max_epoch}")

                fig = plot_training_charts(history, skip_first=skip_first)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Also show parameter scatter plot - use trainer's params if available
                params_to_plot = config['selected_params']
                if st.session_state.trainer:
                    params_to_plot = st.session_state.trainer.selected_params
                param_fig = plot_parameter_scatter(history, params_to_plot, skip_first=skip_first)
                if param_fig:
                    st.plotly_chart(param_fig, use_container_width=True)
    
    if load_checkpoint:
        checkpoint_path = os.path.join(PATHS['checkpoints'], 'best_model.pt')
        if os.path.exists(checkpoint_path):
            with st.spinner("Restoring dataloaders and model architecture..."):
                # 1. Load config to ensure we use the exact parameters it was trained on
                config_path = checkpoint_path.replace('.pt', '_config.json')
                loaded_params = ALL_PARAMETERS
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        ckpt_config = json.load(f)
                    loaded_params = ckpt_config.get('selected_params', ALL_PARAMETERS)
                
                st.write(f"**Model trained on:** {', '.join(loaded_params)}")
                
                try:
                    # 2. Recreate Dataloaders (required to get stats for normalization)
                    dataloaders = create_dataloaders(
                        metadata_csv=PATHS['metadata'],
                        dataset_dir=PATHS['dataset_dir'],
                        selected_params=loaded_params,
                        batch_size=config['batch_size'],
                        num_workers=TRAINING_CONFIG['num_workers'],
                        persistent_workers=TRAINING_CONFIG.get('persistent_workers', True),
                        prefetch_factor=TRAINING_CONFIG.get('prefetch_factor', 2),
                        train_ratio=config['train_ratio'],
                        val_ratio=config['val_ratio'],
                        test_ratio=config['test_ratio']
                    )
                    
                    # 3. Initialize Model Architecture
                    model = create_model(
                        num_params=len(loaded_params),
                        dropout=HYPERPARAMS['dropout'],
                        hidden_dim=HYPERPARAMS['hidden_dim'],
                        device=config['device']
                    )
                    
                    scheduler_type = 'cosine' if 'Cosine' in config['scheduler_choice'] else 'plateau'
                    
                    # 4. Initialize Trainer (bridges the dataloaders, model, and metrics)
                    trainer = create_trainer(
                        model=model,
                        dataloaders=dataloaders,
                        stats=dataloaders['stats'],
                        selected_params=loaded_params,
                        learning_rate=config['learning_rate'],
                        weight_decay=config['weight_decay'],
                        device=config['device'],
                        checkpoint_dir=PATHS['checkpoints'],
                        patience=config['patience'],
                        use_amp=TRAINING_CONFIG.get('use_amp', True),
                        use_bfloat16=TRAINING_CONFIG.get('use_bfloat16', True),
                        compile_model=TRAINING_CONFIG.get('compile_model', True),
                        val_every_n_epochs=TRAINING_CONFIG.get('val_every_n_epochs', 2),
                        use_tf32=TRAINING_CONFIG.get('use_tf32', True),
                        accumulation_steps=TRAINING_CONFIG.get('accumulation_steps', 1),
                        scheduler_type=scheduler_type
                    )

                    # 5. Load the Weights
                    trainer.load_checkpoint(checkpoint_path)
                    
                    # 6. Load Training History Graph (if available)
                    history_path = os.path.join(PATHS['checkpoints'], 'training_history.json')
                    history = None
                    if os.path.exists(history_path):
                        with open(history_path, 'r') as f:
                            full_history = json.load(f)
                        # Downsample history to avoid websocket buffer overflow
                        # Keep first 10 epochs, then sample every 5th epoch
                        history = {}
                        for key, values in full_history.items():
                            if isinstance(values, list) and len(values) > 50:
                                # Keep first 10, then every 5th
                                history[key] = values[:10] + values[10::5]
                            else:
                                history[key] = values
                    
                    # 7. Update Streamlit Session State
                    st.session_state.dataloaders = dataloaders
                    st.session_state.model = model
                    st.session_state.trainer = trainer
                    st.session_state.history = history
                    st.session_state.training_in_progress = False
                    
                    st.success(f"✅ Successfully loaded model from Epoch {trainer.current_epoch}")
                    
                    # 8. Auto-Evaluate to populate the "Results" tab
                    with st.spinner("Running test set evaluation..."):
                        eval_results = trainer.evaluate('test')
                        st.session_state.evaluation_results = eval_results
                        
                except Exception as e:
                    st.error(f"Failed to load checkpoint: {str(e)}")
        else:
            st.error("No checkpoint found! Train a model first.")


def render_meaningful_metrics_table(results, selected_params):
    """
    Render metrics table with meaningful metrics first, then problematic params with both versions.

    Parameters with conditional relevance (decay, shape) are shown at the bottom
    with both full and meaningful metrics including relative error.
    """
    # Separate params into two groups
    meaningful_first_params = []
    problematic_params = []

    for param in selected_params:
        if param in MEANINGFUL_THRESHOLDS:
            problematic_params.append(param)
        else:
            meaningful_first_params.append(param)

    # Build two separate dataframes and concatenate
    regular_data = []
    problematic_data = []

    # First: parameters without filtering (single metrics)
    for param in meaningful_first_params:
        if param in results:
            regular_data.append({
                'Parameter': param,
                'MSE': results[param]['MSE'],
                'MAE': results[param]['MAE'],
                'Rel Error (%)': results[param]['RelativeError_%']
            })

    # Then: parameters with filtering (show both meaningful and full)
    for param in problematic_params:
        if param in results:
            param_results = results[param]

            if 'meaningful' in param_results:
                meaningful = param_results['meaningful']
                row = {
                    'Parameter': f"{param}*",
                    'MSE (meaningful)': meaningful['MSE'],
                    'MSE (full)': param_results['MSE'],
                    'MAE (meaningful)': meaningful['MAE'],
                    'MAE (full)': param_results['MAE'],
                    'Rel Error % (meaningful)': meaningful['RelativeError_%'],
                    'Rel Error % (full)': param_results['RelativeError_%'],
                    '% Excluded': f"{param_results['excluded_pct']:.1f}%"
                }
            else:
                row = {
                    'Parameter': param,
                    'MSE (meaningful)': param_results['MSE'],
                    'MSE (full)': param_results['MSE'],
                    'MAE (meaningful)': param_results['MAE'],
                    'MAE (full)': param_results['MAE'],
                    'Rel Error % (meaningful)': param_results['RelativeError_%'],
                    'Rel Error % (full)': param_results['RelativeError_%'],
                    '% Excluded': '-'
                }
            problematic_data.append(row)

    # Return combined data for display
    return regular_data, problematic_data


def render_results():
    """Render results visualization tab."""
    st.markdown("<div class='section-header'>📈 Results & Evaluation</div>", unsafe_allow_html=True)
    
    # Show live training progress if training is active and we have history
    if st.session_state.training_in_progress and st.session_state.history:
        st.info("⏳ Training in progress... Showing live parameter validation MSE")
        
        history = st.session_state.history
        selected_params = st.session_state.trainer.selected_params if st.session_state.trainer else ALL_PARAMETERS
        
        # Live per-parameter validation MSE chart
        st.subheader("📊 Live Per-Parameter Validation MSE")
        param_fig = plot_parameter_scatter(history, selected_params, skip_first=0)
        if param_fig:
            st.plotly_chart(param_fig, use_container_width=True, key="live_results_param_fig")
        
        # Live parameter table
        if 'per_param_val' in history:
            st.subheader("📋 Current Parameter MSE Values")
            param_data = []
            for param in selected_params:
                if param in history['per_param_val'] and len(history['per_param_val'][param]) > 0:
                    val_values = history['per_param_val'][param]
                    latest_mse = val_values[-1]
                    # Get last validated epoch (find non-inf last value)
                    last_valid_mse = latest_mse
                    for val in reversed(val_values[:-1]):
                        if val != float('inf') and val > 0:
                            last_valid_mse = val
                            break
                    
                    param_data.append({
                        'Parameter': param,
                        'Current MSE': f"{latest_mse:.6f}",
                        'Last Validated': f"{last_valid_mse:.6f}" if last_valid_mse != latest_mse else "Same",
                        'Epochs Tracked': len(val_values)
                    })
            
            if param_data:
                param_df = pd.DataFrame(param_data)
                st.dataframe(param_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.caption("🔄 This tab updates automatically during training. Switch to 'Training' tab to see progress.")
        return
    
    # Final results (after training/evaluation complete)
    if not st.session_state.evaluation_results:
        st.info("No evaluation results yet. Train a model first!")
        return
    
    results = st.session_state.evaluation_results
    selected_params = st.session_state.trainer.selected_params if st.session_state.trainer else ALL_PARAMETERS
    
    # Overall metrics
    st.subheader("Overall Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{results['overall']['MSE']:.6f}")
    with col2:
        st.metric("RMSE", f"{results['overall']['RMSE']:.6f}")
    with col3:
        st.metric("MAE", f"{results['overall']['MAE']:.6f}")
    
    st.markdown("---")

    # Show training history curves if available (for loaded checkpoints)
    if st.session_state.history and not st.session_state.training_in_progress:
        st.subheader("📊 Training History")
        history = st.session_state.history

        if 'train_loss' in history and len(history['train_loss']) > 0:
            max_epoch = len(history['train_loss'])
            default_skip = min(3, max_epoch - 1)

            col1, col2 = st.columns([1, 3])
            with col1:
                skip_first = st.number_input(
                    "Start from epoch",
                    min_value=0,
                    max_value=max_epoch - 1,
                    value=default_skip,
                    step=1,
                    help="Skip initial epochs to better see convergence",
                    key="results_skip_first"  # unique key for this instance
                )
            with col2:
                if skip_first > 0:
                    st.caption(f"📊 Showing epochs {skip_first + 1} to {max_epoch}")

            fig = plot_training_charts(history, skip_first=skip_first)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="results_training_curves")

            param_fig = plot_parameter_scatter(history, selected_params, skip_first=skip_first)
            if param_fig:
                st.plotly_chart(param_fig, use_container_width=True, key="results_param_scatter")

        st.markdown("---")

    # Per-parameter metrics table with meaningful filtering
    st.subheader("Per-Parameter Performance")

    # Show note about conditional relevance
    st.caption("*Parameters with conditional relevance: Decay excluded when sustain > 0.9, Shape excluded when sqr < 0.15")

    regular_data, problematic_data = render_meaningful_metrics_table(results, selected_params)

    # Display regular parameters (single metrics)
    if regular_data:
        regular_df = pd.DataFrame(regular_data)
        st.dataframe(regular_df, use_container_width=True, hide_index=True)

    # Display problematic parameters with both meaningful and full metrics
    if problematic_data:
        st.markdown("**Parameters with Conditional Relevance**")
        problematic_df = pd.DataFrame(problematic_data)
        st.dataframe(problematic_df, use_container_width=True, hide_index=True)
    
    # Manual export button
    export_col1, export_col2 = st.columns([1, 3])
    with export_col1:
        if st.button("📥 Export Metrics to CSV"):
            try:
                results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
                os.makedirs(results_dir, exist_ok=True)
                timestamp = pd.Timestamp.now().strftime('%Y-%m-%dT%H-%M-%S')
                export_path = os.path.join(results_dir, f'{timestamp}_metrics.csv')
                export_metrics_to_csv(results, selected_params, export_path)
                st.success(f"Exported to: {export_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    st.markdown("---")
    
    # Scatter plots
    st.subheader("Prediction vs Ground Truth")
    
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    # Build targets dict for filtering
    targets_dict = {}
    for i, param in enumerate(selected_params):
        targets_dict[param] = targets[:, i]
    
    # Create a responsive 3-column grid
    cols = st.columns(3)
    
    for idx, param in enumerate(selected_params):
        with cols[idx % 3]:
            fig = go.Figure()
            
            # Check if this parameter has conditional relevance
            if param in MEANINGFUL_THRESHOLDS and 'targets_dict' in results:
                # Get meaningful mask
                config = MEANINGFUL_THRESHOLDS[param]
                condition_param = config['condition']
                threshold = config['threshold']
                operator = config['operator']
                
                condition_values = np.array(results['targets_dict'][condition_param])
                
                if operator == 'le':
                    meaningful_mask = condition_values <= threshold
                else:
                    meaningful_mask = condition_values >= threshold
                
                # Plot excluded samples (faded)
                if np.any(~meaningful_mask):
                    fig.add_trace(go.Scatter(
                        x=targets[~meaningful_mask, idx],
                        y=predictions[~meaningful_mask, idx],
                        mode='markers',
                        marker=dict(size=6, opacity=0.15, color='gray'),
                        name='Excluded (irrelevant)'
                    ))
                
                # Plot meaningful samples (full opacity)
                if np.any(meaningful_mask):
                    fig.add_trace(go.Scatter(
                        x=targets[meaningful_mask, idx],
                        y=predictions[meaningful_mask, idx],
                        mode='markers',
                        marker=dict(size=6, opacity=0.6, color='#1f77b4'),
                        name='Meaningful samples'
                    ))
            else:
                # Standard scatter for non-filtered params
                fig.add_trace(go.Scatter(
                    x=targets[:, idx],
                    y=predictions[:, idx],
                    mode='markers',
                    marker=dict(size=6, opacity=0.4, color='#1f77b4'),
                    name='Predictions'
                ))
            
            # Add perfect prediction line
            min_val = min(targets[:, idx].min(), predictions[:, idx].min())
            max_val = max(targets[:, idx].max(), predictions[:, idx].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            ))
            
            # Add subtitle if filtered
            title_text = param.capitalize()
            if param in MEANINGFUL_THRESHOLDS:
                config = MEANINGFUL_THRESHOLDS[param]
                if param == 'decay':
                    title_text += "<br><sub>Gray = sustain > 0.9</sub>"
                elif param == 'shape':
                    title_text += "<br><sub>Gray = sqr < 0.15</sub>"
            
            fig.update_layout(
                xaxis_title='Ground Truth',
                yaxis_title='Predicted',
                title=title_text,
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_inference():
    """Render inference tab with two modes: Upload spectrogram or Generate from SynthAX parameters."""
    st.markdown("<div class='section-header'>🔮 Single Inference</div>", unsafe_allow_html=True)

    if not st.session_state.model or not st.session_state.trainer:
        st.warning("No trained model available. Train or load a model first!")
        return

    # Two modes: Upload or Generate (NO dataset sample option)
    mode = st.radio("Inference Mode", ["Upload Spectrogram", "Generate Audio"], horizontal=True)

    if mode == "Upload Spectrogram":
        render_inference_upload()
    else:
        render_inference_generate()


def render_inference_upload():
    """Mode 1: Upload a spectrogram and predict parameters + synthesized audio/spectrogram."""
    st.subheader("📤 Upload Spectrogram")
    
    uploaded_file = st.file_uploader("Upload a spectrogram image", type=['png', 'jpg', 'jpeg'])
    
    if not uploaded_file:
        st.info("Upload a spectrogram image to proceed.")
        return
    
    if not st.button("Run Prediction", key="btn_predict_upload"):
        return
    
    # Load and prepare image
    image = Image.open(uploaded_file).convert('L')
    img_tensor = prepare_image_tensor(image, device='cpu')
    
    # Run prediction
    try:
        pred_result = predict_and_synthesize(st.session_state.trainer, img_tensor)
        
        # Spectrograms side by side
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Uploaded Spectrogram")
            st.image(image, use_container_width=True)
        with col_right:
            st.subheader("Predicted Spectrogram")
            st.image(pred_result['spec_img'], use_container_width=True)

        # Difference spectrogram centered below
        diff_img = spectrogram_diff_image(image, pred_result['spec_img'])
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            st.subheader("Spectrogram Difference")
            st.image(diff_img, use_container_width=True)

        # Centered parameter tables
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            st.subheader("Predicted Parameters")
            st.markdown("**Decoded Raw Parameters**")
            st.dataframe(params_dict_to_dataframe(pred_result['pred_raw']), use_container_width=True, hide_index=True)
            st.markdown("**Normalized Parameters**")
            norm_table = [
                {'Parameter': p, 'Value': f"{v:.4f}"}
                for p, v in zip(st.session_state.trainer.selected_params, pred_result['pred_norm'])
            ]
            st.dataframe(pd.DataFrame(norm_table), use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")


def render_inference_generate():
    """Mode 2: Generate audio using SynthAX parameters (sliders) → predict from its spectrogram."""
    st.subheader("🎛️ Generate Audio & Predict")
    
    st.markdown("**Set SynthAX Parameters** (use sliders to adjust)")
    
    # Create sliders for all 11 parameters (raw units)
    raw_params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        raw_params['midi_f0'] = st.slider("f0 (MIDI note)", min_value=21.0, max_value=108.0, value=69.0, step=1.0)
        raw_params['cutoff'] = st.slider("Cutoff (Hz)", min_value=100.0, max_value=20000.0, value=1000.0, step=100.0)
        raw_params['attack'] = st.slider("Attack (seconds)", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
        raw_params['decay'] = st.slider("Decay (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        raw_params['sustain'] = st.slider("Sustain", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        raw_params['release'] = st.slider("Release (seconds)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    
    with col2:
        raw_params['alpha'] = st.slider("Alpha", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        raw_params['noise'] = st.slider("Noise", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
        raw_params['sine'] = st.slider("Sine Level", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        raw_params['sqr'] = st.slider("Square Level", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        raw_params['shape'] = st.slider("Shape (Square-Saw)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    use_original_frequency = st.checkbox(
        "Reconstruct predicted audio using original frequency (ignore predicted f0)",
        value=False,
        key="use_original_frequency"
    )
    
    if not st.button("Generate & Predict", key="btn_generate_predict"):
        return
    
    try:
        orig_assets, pred_result = generate_and_predict(raw_params, st.session_state.trainer, device='cpu')

        display_spec_img = pred_result['spec_img']
        display_wav_bytes = pred_result['wav_bytes']
        display_waveform = pred_result['waveform']
        reconstructed_note = ""

        if use_original_frequency:
            recon_pred_raw = dict(pred_result['pred_raw'])
            recon_pred_raw['midi_f0'] = raw_params['midi_f0']
            recon_assets = synthesize_from_raw_params(recon_pred_raw)
            display_spec_img = recon_assets['spec_img']
            display_wav_bytes = recon_assets['wav_bytes']
            display_waveform = recon_assets['waveform']
            reconstructed_note = " (with original MIDI f0)"
        
        # Spectrograms side by side
        col_spec_orig, col_spec_pred = st.columns(2)
        with col_spec_orig:
            st.subheader("Original Spectrogram")
            st.image(orig_assets['spec_img'], use_container_width=True)
        with col_spec_pred:
            st.subheader(f"Predicted Spectrogram{reconstructed_note}")
            st.image(display_spec_img, use_container_width=True)

        # Difference spectrogram centered below
        diff_img = spectrogram_diff_image(orig_assets['spec_img'], display_spec_img)
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            st.subheader("Spectrogram Difference")
            st.image(diff_img, use_container_width=True)

        # Audio players side by side
        col_audio_orig, col_audio_pred = st.columns(2)
        with col_audio_orig:
            st.subheader("Original Audio")
            st.audio(orig_assets['wav_bytes'], format='audio/wav')
            st.download_button(label="Download Original WAV", data=orig_assets['wav_bytes'], file_name="original.wav", mime="audio/wav")
        with col_audio_pred:
            st.subheader(f"Predicted Audio{reconstructed_note}")
            st.audio(display_wav_bytes, format='audio/wav')
            st.download_button(label="Download Predicted WAV", data=display_wav_bytes, file_name="predicted.wav", mime="audio/wav")

        # Difference audio centered below
        diff_audio = waveform_diff_wav_bytes(orig_assets['waveform'], display_waveform)
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            st.subheader("Audio Difference")
            st.audio(diff_audio, format='audio/wav')
            st.download_button(label="Download Difference WAV", data=diff_audio, file_name="difference.wav", mime="audio/wav")

        # Parameter tables: raw originals left, predicted decoded right, normalized under predicted decoded
        col_params_left, col_params_right = st.columns(2)
        with col_params_left:
            st.subheader("Original Parameters")
            st.dataframe(params_dict_to_dataframe(raw_params), use_container_width=True, hide_index=True)

        with col_params_right:
            st.subheader("Predicted Decoded Parameters")
            st.dataframe(params_dict_to_dataframe(pred_result['pred_raw']), use_container_width=True, hide_index=True)
            st.markdown("**Normalized Parameters**")
            norm_table = [
                {'Parameter': p, 'Value': f"{v:.4f}"}
                for p, v in zip(st.session_state.trainer.selected_params, pred_result['pred_norm'])
            ]
            st.dataframe(pd.DataFrame(norm_table), use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Generation/Prediction failed: {e}")


def main():
    """Main application entry point."""
    st.markdown("""
    Train a CNN to predict synthesizer parameters from mel-spectrograms.
    Select parameters, adjust hyperparameters in the sidebar, and start training!
    """)
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Exploration", 
        "🚀 Training", 
        "📈 Results", 
        "🔮 Inference"
    ])
    
    with tab1:
        render_data_exploration()
    
    with tab2:
        render_training_control(config)
    
    with tab3:
        render_results()
    
    with tab4:
        render_inference()
    
    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
