#!/usr/bin/env python3
"""
Setup script for Text Style Transfer project

This script helps set up the project environment and download required models.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install required dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    return True


def download_models():
    """Download required models."""
    models = [
        "facebook/bart-large-cnn",
        "facebook/bart-base",
        "t5-small"
    ]
    
    for model in models:
        if not run_command(
            f"python -c \"from transformers import pipeline; pipeline('text2text-generation', model='{model}')\"",
            f"Downloading {model}"
        ):
            print(f"⚠️  Warning: Failed to download {model}")
    
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["models", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def run_tests():
    """Run the test suite."""
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Warning: Some tests failed")
        return False
    return True


def main():
    """Main setup function."""
    print("🚀 Text Style Transfer Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download models (optional)
    print("\n📥 Downloading models (this may take a while)...")
    download_models()
    
    # Run tests
    print("\n🧪 Running tests...")
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: python demo.py")
    print("2. Launch web interface: streamlit run web_app/app.py")
    print("3. Try CLI: python cli.py single 'Your text here' --target-style casual")
    print("4. Read the README.md for more information")


if __name__ == "__main__":
    main()
