#!/usr/bin/env python3
"""
Demo script for Text Style Transfer

This script demonstrates the basic functionality of the text style transfer system.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from style_transfer import StyleTransferModel, StyleTransferConfig


def main():
    """Run the demo."""
    print("🔄 Text Style Transfer Demo")
    print("=" * 50)
    
    # Initialize model with default configuration
    print("Loading model...")
    config = StyleTransferConfig()
    model = StyleTransferModel(config)
    
    # Example texts
    examples = [
        "It is with great pleasure that I write to inform you of our upcoming meeting.",
        "I would like to express my sincere gratitude for your assistance.",
        "Please be advised that the deadline has been extended.",
        "We regret to inform you that your application has been rejected."
    ]
    
    print("\nTransforming texts from formal to casual style:")
    print("-" * 50)
    
    for i, text in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Original: {text}")
        
        result = model.transfer_style(text, target_style="casual")
        print(f"Casual: {result['transformed_text']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    print("\n" + "=" * 50)
    print("Demo completed! Try the web interface with: streamlit run web_app/app.py")
    print("Or use the CLI: python cli.py single 'Your text here' --target-style casual")


if __name__ == "__main__":
    main()
