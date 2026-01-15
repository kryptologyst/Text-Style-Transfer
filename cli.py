"""
Command Line Interface for Text Style Transfer

This module provides a CLI for the text style transfer system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from style_transfer import StyleTransferModel, StyleTransferConfig, StyleTransferEvaluator
from data_loader import StyleTransferDataLoader


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('style_transfer.log')
        ]
    )


def single_text_transfer(args) -> None:
    """Handle single text transfer."""
    config = StyleTransferConfig(
        model_name=args.model,
        max_length=args.max_length,
        temperature=args.temperature,
        num_beams=args.num_beams
    )
    
    model = StyleTransferModel(config)
    
    print(f"Transforming text: '{args.text}'")
    print(f"Target style: {args.target_style}")
    print("-" * 50)
    
    result = model.transfer_style(
        args.text,
        target_style=args.target_style,
        source_style=args.source_style
    )
    
    print(f"Original: {result['original_text']}")
    print(f"Transformed: {result['transformed_text']}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")


def batch_transfer(args) -> None:
    """Handle batch text transfer."""
    config = StyleTransferConfig(
        model_name=args.model,
        max_length=args.max_length,
        temperature=args.temperature,
        num_beams=args.num_beams
    )
    
    model = StyleTransferModel(config)
    
    # Load texts
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = args.texts
    
    print(f"Processing {len(texts)} texts...")
    print("-" * 50)
    
    results = model.batch_transfer(
        texts,
        target_style=args.target_style,
        source_style=args.source_style
    )
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nText {i}:")
        print(f"Original: {result['original_text']}")
        print(f"Transformed: {result['transformed_text']}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
    
    # Save results if output file specified
    if args.output_file:
        import json
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_file}")


def evaluate_model(args) -> None:
    """Handle model evaluation."""
    config = StyleTransferConfig(
        model_name=args.model,
        max_length=args.max_length,
        temperature=args.temperature,
        num_beams=args.num_beams
    )
    
    model = StyleTransferModel(config)
    evaluator = StyleTransferEvaluator()
    
    # Load dataset
    data_loader = StyleTransferDataLoader(args.dataset)
    
    # Get appropriate pairs based on evaluation type
    if args.eval_type == "formal_casual":
        pairs = data_loader.get_formal_casual_pairs()
        source_style, target_style = "formal", "casual"
    elif args.eval_type == "sentiment":
        pairs = data_loader.get_sentiment_pairs()
        source_style, target_style = "negative", "positive"
    else:
        pairs = data_loader.get_professional_friendly_pairs()
        source_style, target_style = "professional", "friendly"
    
    print(f"Evaluating {args.eval_type} transfer...")
    print(f"Using {len(pairs)} pairs from dataset")
    print("-" * 50)
    
    # Process pairs
    results = []
    reference_texts = []
    
    for source, target, category in pairs[:args.max_samples]:
        result = model.transfer_style(source, target_style, source_style)
        results.append(result)
        reference_texts.append(target)
    
    # Evaluate
    metrics = evaluator.evaluate_transformations(results, reference_texts)
    
    print("Evaluation Results:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")


def dataset_info(args) -> None:
    """Display dataset information."""
    data_loader = StyleTransferDataLoader(args.dataset)
    all_pairs = data_loader.get_all_pairs()
    
    print("Dataset Information:")
    print("=" * 50)
    
    total_pairs = 0
    for transfer_type, pairs in all_pairs.items():
        print(f"{transfer_type.replace('_', ' ').title()}: {len(pairs)} pairs")
        total_pairs += len(pairs)
    
    print(f"\nTotal pairs: {total_pairs}")
    
    # Show sample data
    if args.show_samples:
        print("\nSample Data:")
        print("-" * 30)
        
        for transfer_type, pairs in all_pairs.items():
            if pairs:
                print(f"\n{transfer_type.replace('_', ' ').title()}:")
                source, target, category = pairs[0]
                print(f"  Source: {source}")
                print(f"  Target: {target}")
                print(f"  Category: {category}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Text Style Transfer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text transfer
  python cli.py single "It is with great pleasure" --target-style casual
  
  # Batch transfer from file
  python cli.py batch --input-file texts.txt --output-file results.json
  
  # Evaluate model
  python cli.py evaluate --eval-type formal_casual --max-samples 10
  
  # Show dataset info
  python cli.py dataset-info --show-samples
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single text transfer
    single_parser = subparsers.add_parser("single", help="Transform single text")
    single_parser.add_argument("text", help="Text to transform")
    single_parser.add_argument("--target-style", default="casual", help="Target style")
    single_parser.add_argument("--source-style", default="formal", help="Source style")
    single_parser.add_argument("--model", default="facebook/bart-large-cnn", help="Model to use")
    single_parser.add_argument("--max-length", type=int, default=512, help="Max output length")
    single_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    single_parser.add_argument("--num-beams", type=int, default=4, help="Number of beams")
    
    # Batch transfer
    batch_parser = subparsers.add_parser("batch", help="Transform multiple texts")
    batch_group = batch_parser.add_mutually_exclusive_group(required=True)
    batch_group.add_argument("--texts", nargs="+", help="Texts to transform")
    batch_group.add_argument("--input-file", help="File containing texts (one per line)")
    batch_parser.add_argument("--output-file", help="Output file for results")
    batch_parser.add_argument("--target-style", default="casual", help="Target style")
    batch_parser.add_argument("--source-style", default="formal", help="Source style")
    batch_parser.add_argument("--model", default="facebook/bart-large-cnn", help="Model to use")
    batch_parser.add_argument("--max-length", type=int, default=512, help="Max output length")
    batch_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    batch_parser.add_argument("--num-beams", type=int, default=4, help="Number of beams")
    
    # Evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--eval-type", choices=["formal_casual", "sentiment", "professional_friendly"], 
                           default="formal_casual", help="Type of evaluation")
    eval_parser.add_argument("--dataset", default="data/synthetic_dataset.json", help="Dataset path")
    eval_parser.add_argument("--max-samples", type=int, default=20, help="Maximum samples to evaluate")
    eval_parser.add_argument("--model", default="facebook/bart-large-cnn", help="Model to use")
    eval_parser.add_argument("--max-length", type=int, default=512, help="Max output length")
    eval_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    eval_parser.add_argument("--num-beams", type=int, default=4, help="Number of beams")
    
    # Dataset info
    dataset_parser = subparsers.add_parser("dataset-info", help="Show dataset information")
    dataset_parser.add_argument("--dataset", default="data/synthetic_dataset.json", help="Dataset path")
    dataset_parser.add_argument("--show-samples", action="store_true", help="Show sample data")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "single":
        single_text_transfer(args)
    elif args.command == "batch":
        batch_transfer(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "dataset-info":
        dataset_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
