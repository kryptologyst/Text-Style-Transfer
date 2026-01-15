"""
Text Style Transfer Module

This module provides functionality for transferring text styles using various
transformer models and techniques. It supports multiple style transfer approaches
including fine-tuned models, zero-shot methods, and few-shot learning.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from datasets import Dataset
import evaluate
from sklearn.metrics import accuracy_score, f1_score
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StyleTransferConfig:
    """Configuration class for style transfer parameters."""
    
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 4
    early_stopping: bool = True
    device: str = "auto"
    batch_size: int = 1


class StyleTransferModel:
    """
    A comprehensive style transfer model supporting multiple approaches.
    
    This class provides a unified interface for different style transfer methods
    including BART-based models, T5 models, and custom fine-tuned approaches.
    """
    
    def __init__(self, config: StyleTransferConfig):
        """
        Initialize the style transfer model.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Load model based on architecture
            if "bart" in self.config.model_name.lower():
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name
                )
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device
                )
            elif "t5" in self.config.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name
                )
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            else:
                # Fallback to generic model loading
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name
                )
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device
                )
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def transfer_style(
        self,
        text: str,
        target_style: str = "casual",
        source_style: str = "formal"
    ) -> Dict[str, Union[str, float]]:
        """
        Transfer the style of input text.
        
        Args:
            text: Input text to transform
            target_style: Target style (e.g., 'casual', 'formal', 'positive')
            source_style: Source style (e.g., 'formal', 'casual', 'negative')
            
        Returns:
            Dictionary containing transformed text and metadata
        """
        try:
            if self.pipeline is not None:
                # Use pipeline for BART-based models
                result = self.pipeline(
                    text,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_beams=self.config.num_beams,
                    early_stopping=self.config.early_stopping,
                    do_sample=True
                )
                
                transformed_text = result[0]['generated_text']
                
            elif "t5" in self.config.model_name.lower():
                # Use T5-specific approach
                transformed_text = self._t5_style_transfer(text, target_style)
            
            else:
                # Fallback method
                transformed_text = self._generic_style_transfer(text, target_style)
            
            return {
                "original_text": text,
                "transformed_text": transformed_text,
                "source_style": source_style,
                "target_style": target_style,
                "model_name": self.config.model_name,
                "confidence": self._calculate_confidence(text, transformed_text)
            }
            
        except Exception as e:
            logger.error(f"Error in style transfer: {e}")
            return {
                "original_text": text,
                "transformed_text": text,  # Return original on error
                "source_style": source_style,
                "target_style": target_style,
                "model_name": self.config.model_name,
                "error": str(e)
            }
    
    def _t5_style_transfer(self, text: str, target_style: str) -> str:
        """T5-specific style transfer implementation."""
        # Create style-specific prompt
        prompt = f"Convert to {target_style} style: {text}"
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping,
                do_sample=True
            )
        
        # Decode output
        transformed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return transformed_text
    
    def _generic_style_transfer(self, text: str, target_style: str) -> str:
        """Generic style transfer fallback method."""
        # Simple rule-based transformation as fallback
        transformations = {
            "casual": {
                "It is with great pleasure": "I'm excited to",
                "I would like to inform you": "Just wanted to let you know",
                "Please be advised": "FYI",
                "at your earliest convenience": "whenever you can"
            },
            "formal": {
                "I'm excited to": "It is with great pleasure",
                "Just wanted to let you know": "I would like to inform you",
                "FYI": "Please be advised",
                "whenever you can": "at your earliest convenience"
            }
        }
        
        transformed_text = text
        if target_style in transformations:
            for old, new in transformations[target_style].items():
                transformed_text = transformed_text.replace(old, new)
        
        return transformed_text
    
    def _calculate_confidence(self, original: str, transformed: str) -> float:
        """Calculate confidence score for the transformation."""
        # Simple heuristic: confidence based on text length preservation
        length_ratio = len(transformed) / len(original) if len(original) > 0 else 0
        return min(1.0, max(0.0, 1.0 - abs(1.0 - length_ratio)))
    
    def batch_transfer(
        self,
        texts: List[str],
        target_style: str = "casual",
        source_style: str = "formal"
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Perform style transfer on a batch of texts.
        
        Args:
            texts: List of input texts
            target_style: Target style for all texts
            source_style: Source style for all texts
            
        Returns:
            List of transformation results
        """
        results = []
        for text in texts:
            result = self.transfer_style(text, target_style, source_style)
            results.append(result)
        return results


class StyleTransferEvaluator:
    """Evaluator for style transfer models."""
    
    def __init__(self):
        """Initialize the evaluator with metrics."""
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
    
    def evaluate_transformations(
        self,
        results: List[Dict[str, Union[str, float]]],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate style transfer results.
        
        Args:
            results: List of transformation results
            reference_texts: Optional reference texts for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not results:
            return {}
        
        # Extract predictions and references
        predictions = [r["transformed_text"] for r in results]
        originals = [r["original_text"] for r in results]
        
        # Calculate metrics
        metrics = {}
        
        # BLEU score (if references provided)
        if reference_texts:
            bleu_scores = []
            for pred, ref in zip(predictions, reference_texts):
                bleu_score = self.bleu_metric.compute(
                    predictions=[pred],
                    references=[[ref]]
                )["bleu"]
                bleu_scores.append(bleu_score)
            metrics["bleu"] = np.mean(bleu_scores)
        
        # ROUGE score
        rouge_scores = self.rouge_metric.compute(
            predictions=predictions,
            references=originals
        )
        metrics.update({
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"]
        })
        
        # Average confidence
        confidences = [r.get("confidence", 0.0) for r in results]
        metrics["avg_confidence"] = np.mean(confidences)
        
        return metrics


def load_config(config_path: str) -> StyleTransferConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return StyleTransferConfig(**config_dict)


def main():
    """Main function demonstrating style transfer capabilities."""
    # Load configuration
    config = StyleTransferConfig()
    
    # Initialize model
    model = StyleTransferModel(config)
    
    # Example texts for demonstration
    example_texts = [
        "It is with great pleasure that I write to inform you of our upcoming meeting.",
        "I would like to express my sincere gratitude for your assistance.",
        "Please be advised that the deadline has been extended.",
        "We regret to inform you that your application has been rejected."
    ]
    
    print("Text Style Transfer Demo")
    print("=" * 50)
    
    # Perform style transfers
    for i, text in enumerate(example_texts, 1):
        print(f"\nExample {i}:")
        print(f"Original: {text}")
        
        result = model.transfer_style(text, target_style="casual")
        print(f"Casual: {result['transformed_text']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    # Batch processing example
    print(f"\n{'='*50}")
    print("Batch Processing Example:")
    batch_results = model.batch_transfer(example_texts[:2], target_style="casual")
    
    # Evaluate results
    evaluator = StyleTransferEvaluator()
    metrics = evaluator.evaluate_transformations(batch_results)
    
    print(f"\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")


if __name__ == "__main__":
    main()
