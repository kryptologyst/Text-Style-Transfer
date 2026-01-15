#!/usr/bin/env python3
"""
Simple test script for Text Style Transfer

This script tests the basic functionality without requiring model downloads.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from style_transfer import StyleTransferConfig
from data_loader import StyleTransferDataLoader


def test_data_loader():
    """Test the data loader functionality."""
    print("🧪 Testing Data Loader...")
    
    try:
        loader = StyleTransferDataLoader("data/synthetic_dataset.json")
        all_pairs = loader.get_all_pairs()
        
        print(f"✅ Loaded dataset with {sum(len(pairs) for pairs in all_pairs.values())} pairs")
        
        for transfer_type, pairs in all_pairs.items():
            print(f"  - {transfer_type}: {len(pairs)} pairs")
        
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False


def test_config():
    """Test the configuration system."""
    print("\n🧪 Testing Configuration...")
    
    try:
        config = StyleTransferConfig()
        print(f"✅ Default model: {config.model_name}")
        print(f"✅ Max length: {config.max_length}")
        print(f"✅ Temperature: {config.temperature}")
        
        # Test custom config
        custom_config = StyleTransferConfig(model_name="t5-small", max_length=256)
        print(f"✅ Custom model: {custom_config.model_name}")
        print(f"✅ Custom max length: {custom_config.max_length}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_generic_style_transfer():
    """Test the generic style transfer fallback."""
    print("\n🧪 Testing Generic Style Transfer...")
    
    try:
        from style_transfer import StyleTransferModel
        
        config = StyleTransferConfig(model_name="t5-small")
        model = StyleTransferModel(config)
        
        # Test generic transformation
        test_text = "It is with great pleasure that I inform you"
        result = model._generic_style_transfer(test_text, "casual")
        
        print(f"✅ Original: {test_text}")
        print(f"✅ Transformed: {result}")
        
        # Test confidence calculation
        confidence = model._calculate_confidence(test_text, result)
        print(f"✅ Confidence: {confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Generic style transfer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🔄 Text Style Transfer - Basic Tests")
    print("=" * 50)
    
    tests = [
        test_data_loader,
        test_config,
        test_generic_style_transfer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full demo: python demo.py")
        print("3. Launch web interface: streamlit run web_app/app.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()
