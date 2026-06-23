#!/usr/bin/env python3
"""
Entry point to launch the SynthAX CNN Parameter Predictor UI.

Usage:
    python run.py

Or directly with Streamlit:
    streamlit run ui/app.py
"""

import subprocess
import sys
import os


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import streamlit
        import pandas
        import plotly
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install requirements:")
        print("    pip install -r requirements.txt")
        return False


def main():
    """Launch the Streamlit application."""
    # Check we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check dataset exists
    from config import PATHS
    if not os.path.exists(PATHS['metadata']):
        print(f"\n⚠️  Warning: Dataset not found at {PATHS['metadata']}")
        print("Make sure the SynthDataset folder exists with metadata.csv and spectrograms/")
    
    print("\n" + "="*60)
    print("🎹 Starting SynthAX CNN Parameter Predictor")
    print("="*60 + "\n")
    
    # Launch Streamlit
    app_path = os.path.join("ui", "app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port=8501",
        "--server.headless=true",
        "--browser.serverAddress=localhost"
    ]
    
    print("Launching Streamlit UI...")
    print("The app will open in your browser at http://localhost:8501\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
