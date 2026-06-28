# Modular Synth Modeling and Parameter Estimation with SYNTHAX

A deep learning system for modeling synthesizer parameters from audio spectrograms using PyTorch and the SYNTHAX audio synthesis framework.

## Project Structure

### Root-Level Scripts

- **`synthax_batch_generator.py`** — Generates batches of synthesized audio samples with random parameters using SYNTHAX
- **`synthax_synth_randomized.py`** — Creates individual randomized synthesized audio files for experimentation
- **`resume_generator.py`** — Utility for resuming interrupted dataset generation processes
- **`clean_dataset.py`** — Data cleaning and preprocessing pipeline for the SynthDataset
- **`inspect_dataset.py`** — Inspection and analysis tools to examine dataset statistics and metadata

### CNN Predictor Module (`cnn_predictor/`)

- **`config.py`** — Central configuration file with dataset paths, hyperparameters, device settings, and parameter definitions
- **`run.py`** — Main entry point for training and inference
- **`inference_utils.py`** — Model inference utilities and audio synthesis pipeline for parameter prediction and waveform generation
- **`README.md`** — CNN-specific documentation

#### Submodules

- **`models/`** — CNN architecture
  - `cnn_regressor.py` — PyTorch CNN model for predicting 11 synthesizer parameters from spectrograms
  
- **`training/`** — Training infrastructure
  - `train.py` — Training orchestration with mixed precision, early stopping, and checkpointing
  - `utils.py` — Training utilities (loss functions, metrics, schedulers)
  
- **`data/`** — Dataset handling
  - `dataset.py` — PyTorch DataLoader and preprocessing for spectrograms and metadata
  
- **`ui/`** — Interactive interface
  - `app.py` — Streamlit dashboard for model training, evaluation, and real-time inference
  
- **`checkpoints/`** — Trained model weights
  - `best_model.pt` / `best_model_config.json` — Best performing checkpoint
  - `final_model.pt` / `final_model_config.json` — Final trained model
  - `training_history.json` — Training metrics and loss history
  
- **`results/`** — Evaluation outputs
  - CSV files with inference metrics and predictions

### Dataset (`SynthDataset/`)

- **`metadata.csv`** — Metadata for 50,000+ synthesized audio samples with 11 parameter values and spectrogram paths
- **`audio/`** — Raw WAV files from SYNTHAX synthesis engine
- **`spectrograms/`** — Precomputed mel-spectrograms (50,000+ JPG images) used as CNN input

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate Synthetic Data

```bash
python synthax_batch_generator.py
```

### Train the CNN Model
```bash
cd cnn_predictor
python run.py --mode train
```

### Launch Interactive UI
```bash
streamlit run ui/app.py
```

### Make Predictions
Use the UI dashboard or programmatically:
```bash
from cnn_predictor.inference_utils import predict_and_synthesize
audio = predict_and_synthesize(spectrogram_input)
```

## Key Features
- CNN Regressor — Predicts 11 synthesizer parameters from spectrograms
- Mixed Precision Training — bfloat16 optimization with torch.compile for faster convergence
- SYNTHAX Integration — Real-time audio synthesis from predicted parameters
- Interactive Dashboard — Streamlit UI for model evaluation and inference
- Checkpoint Management — Automatic model checkpointing with best and final weights
- Scalable Dataset — 50,000+ synthesized samples with metadata and precomputed spectrograms

## Requirements
- Python 3.10+
- PyTorch 2.0.0+
- CUDA-capable GPU (optional, CPU supported)

See `requirements.txt` for full dependencies.

## Model Output
The CNN predicts 11 synthesizer parameters defined in `config.py`:
- Frequency components
- Amplitude/envelope controls
- Filter characteristics
- Modulation parameters

And more...
