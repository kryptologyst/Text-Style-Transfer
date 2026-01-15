"""
Data loading and preprocessing utilities for text style transfer.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd

logger = logging.getLogger(__name__)


class StyleTransferDataLoader:
    """Data loader for style transfer datasets."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the dataset file
        """
        self.data_path = Path(data_path)
        self.data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Loaded dataset from {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_formal_casual_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Get formal to casual text pairs.
        
        Returns:
            List of tuples (formal_text, casual_text, category)
        """
        pairs = []
        if "formal_to_casual" in self.data:
            for item in self.data["formal_to_casual"]:
                pairs.append((
                    item["formal"],
                    item["casual"],
                    item["category"]
                ))
        return pairs
    
    def get_casual_formal_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Get casual to formal text pairs.
        
        Returns:
            List of tuples (casual_text, formal_text, category)
        """
        pairs = []
        if "casual_to_formal" in self.data:
            for item in self.data["casual_to_formal"]:
                pairs.append((
                    item["casual"],
                    item["formal"],
                    item["category"]
                ))
        return pairs
    
    def get_sentiment_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Get sentiment transfer pairs.
        
        Returns:
            List of tuples (negative_text, positive_text, category)
        """
        pairs = []
        if "sentiment_transfer" in self.data:
            for item in self.data["sentiment_transfer"]:
                pairs.append((
                    item["negative"],
                    item["positive"],
                    item["category"]
                ))
        return pairs
    
    def get_professional_friendly_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Get professional to friendly text pairs.
        
        Returns:
            List of tuples (professional_text, friendly_text, category)
        """
        pairs = []
        if "professional_to_friendly" in self.data:
            for item in self.data["professional_to_friendly"]:
                pairs.append((
                    item["professional"],
                    item["friendly"],
                    item["category"]
                ))
        return pairs
    
    def get_all_pairs(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Get all available text pairs organized by transfer type.
        
        Returns:
            Dictionary mapping transfer types to lists of pairs
        """
        return {
            "formal_to_casual": self.get_formal_casual_pairs(),
            "casual_to_formal": self.get_casual_formal_pairs(),
            "sentiment_transfer": self.get_sentiment_pairs(),
            "professional_to_friendly": self.get_professional_friendly_pairs()
        }
    
    def create_huggingface_dataset(self) -> DatasetDict:
        """
        Create a Hugging Face Dataset from the loaded data.
        
        Returns:
            DatasetDict containing train/validation splits
        """
        all_pairs = self.get_all_pairs()
        
        # Flatten all pairs into a single dataset
        texts = []
        targets = []
        categories = []
        transfer_types = []
        
        for transfer_type, pairs in all_pairs.items():
            for source, target, category in pairs:
                texts.append(source)
                targets.append(target)
                categories.append(category)
                transfer_types.append(transfer_type)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'target': targets,
            'category': categories,
            'transfer_type': transfer_types
        })
        
        # Split into train/validation (80/20)
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        # Create Dataset objects
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def get_sample_texts(self, n: int = 5) -> List[str]:
        """
        Get sample texts for demonstration.
        
        Args:
            n: Number of sample texts to return
            
        Returns:
            List of sample texts
        """
        all_pairs = self.get_all_pairs()
        sample_texts = []
        
        for transfer_type, pairs in all_pairs.items():
            for source, _, _ in pairs[:n//len(all_pairs) + 1]:
                sample_texts.append(source)
                if len(sample_texts) >= n:
                    break
            if len(sample_texts) >= n:
                break
        
        return sample_texts[:n]


def create_mock_dataset(output_path: str) -> None:
    """
    Create a mock dataset for testing purposes.
    
    Args:
        output_path: Path where to save the mock dataset
    """
    mock_data = {
        "formal_to_casual": [
            {
                "formal": "It is imperative that we address this matter immediately.",
                "casual": "We really need to deal with this right away.",
                "category": "urgent_request"
            },
            {
                "formal": "I would be grateful if you could provide your feedback.",
                "casual": "I'd love to hear what you think!",
                "category": "feedback_request"
            }
        ],
        "casual_to_formal": [
            {
                "casual": "This is totally awesome!",
                "formal": "This is exceptionally impressive.",
                "category": "praise"
            },
            {
                "casual": "Can't wait to see what happens next!",
                "formal": "I am eagerly anticipating the next developments.",
                "category": "anticipation"
            }
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created mock dataset at {output_path}")


if __name__ == "__main__":
    # Test the data loader
    data_loader = StyleTransferDataLoader("data/synthetic_dataset.json")
    
    print("Dataset Statistics:")
    print("=" * 50)
    
    all_pairs = data_loader.get_all_pairs()
    for transfer_type, pairs in all_pairs.items():
        print(f"{transfer_type}: {len(pairs)} pairs")
    
    print(f"\nTotal pairs: {sum(len(pairs) for pairs in all_pairs.values())}")
    
    print("\nSample texts:")
    samples = data_loader.get_sample_texts(3)
    for i, text in enumerate(samples, 1):
        print(f"{i}. {text}")
