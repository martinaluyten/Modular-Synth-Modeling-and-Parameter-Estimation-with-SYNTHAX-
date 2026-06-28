# Modular Synth Modeling and Parameter Estimation with SYNTHAX

A deep learning system for modeling synthesizer parameters from audio spectrograms using PyTorch and the SYNTHAX audio synthesis framework.

## Project Structure

- `cnn_predictor/` — CNN model for parameter estimation from spectrograms
  - `models/` — PyTorch CNN architecture
  - `training/` — Training pipeline with mixed precision and early stopping
  - `data/` — Dataset loading and preprocessing
  - `ui/` — Streamlit dashboard for interactive inference
  - `checkpoints/` — Trained model weights
  
- `SynthDataset/` — 50,000+ synthesized audio samples with spectrograms and metadata

- Root Python scripts for dataset generation and processing

## Quick Start

```bash
pip install -r requirements.txt

# Train the model
cd cnn_predictor
python run.py --mode train

# Launch interactive UI
streamlit run ui/app.py
