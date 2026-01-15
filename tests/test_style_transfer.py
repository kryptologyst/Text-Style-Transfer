"""
Test suite for text style transfer functionality.
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from style_transfer import StyleTransferModel, StyleTransferConfig, StyleTransferEvaluator
from data_loader import StyleTransferDataLoader, create_mock_dataset


class TestStyleTransferConfig(unittest.TestCase):
    """Test cases for StyleTransferConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StyleTransferConfig()
        
        self.assertEqual(config.model_name, "facebook/bart-large-cnn")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.num_beams, 4)
        self.assertTrue(config.early_stopping)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.batch_size, 1)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StyleTransferConfig(
            model_name="t5-small",
            max_length=256,
            temperature=0.5
        )
        
        self.assertEqual(config.model_name, "t5-small")
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.temperature, 0.5)


class TestStyleTransferModel(unittest.TestCase):
    """Test cases for StyleTransferModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = StyleTransferConfig(model_name="t5-small")
        # Note: In a real test environment, you might want to use a smaller model
        # or mock the model loading to avoid downloading large models during tests
    
    def test_generic_style_transfer(self):
        """Test the generic style transfer fallback method."""
        model = StyleTransferModel(self.config)
        
        # Test casual transformation
        result = model._generic_style_transfer(
            "It is with great pleasure that I inform you",
            "casual"
        )
        self.assertIn("I'm excited to", result)
        
        # Test formal transformation
        result = model._generic_style_transfer(
            "I'm excited to let you know",
            "formal"
        )
        self.assertIn("It is with great pleasure", result)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        model = StyleTransferModel(self.config)
        
        # Test with similar lengths
        confidence = model._calculate_confidence("Hello world", "Hi there")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with very different lengths
        confidence = model._calculate_confidence("Hi", "This is a very long transformed text")
        self.assertLess(confidence, 1.0)


class TestStyleTransferEvaluator(unittest.TestCase):
    """Test cases for StyleTransferEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = StyleTransferEvaluator()
        self.sample_results = [
            {
                "original_text": "Hello world",
                "transformed_text": "Hi there",
                "confidence": 0.8
            },
            {
                "original_text": "Good morning",
                "transformed_text": "Morning!",
                "confidence": 0.9
            }
        ]
    
    def test_evaluate_transformations(self):
        """Test evaluation of transformations."""
        metrics = self.evaluator.evaluate_transformations(self.sample_results)
        
        self.assertIn("rouge1", metrics)
        self.assertIn("rouge2", metrics)
        self.assertIn("rougeL", metrics)
        self.assertIn("avg_confidence", metrics)
        
        self.assertGreaterEqual(metrics["avg_confidence"], 0.0)
        self.assertLessEqual(metrics["avg_confidence"], 1.0)
    
    def test_evaluate_with_references(self):
        """Test evaluation with reference texts."""
        reference_texts = ["Hi there", "Morning!"]
        metrics = self.evaluator.evaluate_transformations(
            self.sample_results,
            reference_texts
        )
        
        self.assertIn("bleu", metrics)
        self.assertGreaterEqual(metrics["bleu"], 0.0)
        self.assertLessEqual(metrics["bleu"], 1.0)


class TestStyleTransferDataLoader(unittest.TestCase):
    """Test cases for StyleTransferDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test dataset
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.test_data = {
            "formal_to_casual": [
                {
                    "formal": "It is with great pleasure",
                    "casual": "I'm excited to",
                    "category": "greeting"
                }
            ],
            "casual_to_formal": [
                {
                    "casual": "Hey there",
                    "formal": "Good day",
                    "category": "greeting"
                }
            ]
        }
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        Path(self.temp_file.name).unlink()
    
    def test_data_loading(self):
        """Test data loading functionality."""
        loader = StyleTransferDataLoader(self.temp_file.name)
        
        self.assertIsNotNone(loader.data)
        self.assertIn("formal_to_casual", loader.data)
        self.assertIn("casual_to_formal", loader.data)
    
    def test_get_formal_casual_pairs(self):
        """Test getting formal to casual pairs."""
        loader = StyleTransferDataLoader(self.temp_file.name)
        pairs = loader.get_formal_casual_pairs()
        
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "It is with great pleasure")
        self.assertEqual(pairs[0][1], "I'm excited to")
        self.assertEqual(pairs[0][2], "greeting")
    
    def test_get_all_pairs(self):
        """Test getting all pairs."""
        loader = StyleTransferDataLoader(self.temp_file.name)
        all_pairs = loader.get_all_pairs()
        
        self.assertIn("formal_to_casual", all_pairs)
        self.assertIn("casual_to_formal", all_pairs)
        self.assertEqual(len(all_pairs["formal_to_casual"]), 1)
        self.assertEqual(len(all_pairs["casual_to_formal"]), 1)
    
    def test_get_sample_texts(self):
        """Test getting sample texts."""
        loader = StyleTransferDataLoader(self.temp_file.name)
        samples = loader.get_sample_texts(2)
        
        self.assertEqual(len(samples), 2)
        self.assertIn("It is with great pleasure", samples)
        self.assertIn("Hey there", samples)


class TestMockDatasetCreation(unittest.TestCase):
    """Test cases for mock dataset creation."""
    
    def test_create_mock_dataset(self):
        """Test mock dataset creation."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()
        
        try:
            create_mock_dataset(temp_file.name)
            
            # Verify file was created and contains valid JSON
            with open(temp_file.name, 'r') as f:
                data = json.load(f)
            
            self.assertIn("formal_to_casual", data)
            self.assertIn("casual_to_formal", data)
            self.assertGreater(len(data["formal_to_casual"]), 0)
            self.assertGreater(len(data["casual_to_formal"]), 0)
            
        finally:
            Path(temp_file.name).unlink()


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create mock dataset
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()
        
        try:
            create_mock_dataset(temp_file.name)
            
            # Load dataset
            loader = StyleTransferDataLoader(temp_file.name)
            pairs = loader.get_formal_casual_pairs()
            
            # Test with generic style transfer (no model loading)
            config = StyleTransferConfig(model_name="t5-small")
            model = StyleTransferModel(config)
            
            # Test transformation
            if pairs:
                source_text = pairs[0][0]
                result = model._generic_style_transfer(source_text, "casual")
                
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
            
        finally:
            Path(temp_file.name).unlink()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
