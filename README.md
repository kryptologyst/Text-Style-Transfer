# Text Style Transfer

A comprehensive text style transfer system using state-of-the-art transformer models. This project provides multiple interfaces (CLI, Web UI, Python API) for transforming text styles including formal/casual conversion, sentiment transfer, and professional/friendly tone adjustment.

## Features

- **Multiple Style Transfer Types**: Formal ↔ Casual, Sentiment Transfer, Professional ↔ Friendly
- **State-of-the-art Models**: BART, T5, and other transformer models
- **Multiple Interfaces**: CLI, Streamlit Web UI, Python API
- **Comprehensive Evaluation**: BLEU, ROUGE, confidence metrics
- **Visualization**: Interactive charts and performance metrics
- **Testing Suite**: Comprehensive unit tests and integration tests
- **Synthetic Dataset**: Ready-to-use dataset for testing and evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Text-Style-Transfer.git
cd Text-Style-Transfer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models (optional, will download automatically on first use):
```bash
python -c "from transformers import pipeline; pipeline('text2text-generation', model='facebook/bart-large-cnn')"
```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web interface:
```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` and start transforming text styles interactively!

### Command Line Interface

Transform a single text:
```bash
python cli.py single "It is with great pleasure that I inform you" --target-style casual
```

Batch processing:
```bash
python cli.py batch --input-file texts.txt --output-file results.json
```

Evaluate model performance:
```bash
python cli.py evaluate --eval-type formal_casual --max-samples 10
```

### Python API

```python
from src.style_transfer import StyleTransferModel, StyleTransferConfig

# Initialize model
config = StyleTransferConfig(model_name="facebook/bart-large-cnn")
model = StyleTransferModel(config)

# Transform text
result = model.transfer_style(
    "It is with great pleasure that I inform you",
    target_style="casual",
    source_style="formal"
)

print(f"Original: {result['original_text']}")
print(f"Transformed: {result['transformed_text']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Project Structure

```
text-style-transfer/
├── src/                    # Source code
│   ├── style_transfer.py  # Core style transfer functionality
│   └── data_loader.py     # Data loading utilities
├── web_app/               # Streamlit web interface
│   └── app.py            # Main web application
├── data/                  # Datasets and sample data
│   └── synthetic_dataset.json
├── tests/                 # Test suite
│   └── test_style_transfer.py
├── config/                # Configuration files
│   └── config.yaml
├── models/                # Saved models (created at runtime)
├── cli.py                 # Command line interface
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Available Models

- **facebook/bart-large-cnn**: High-quality summarization and style transfer
- **facebook/bart-base**: Smaller, faster BART model
- **t5-small**: Efficient T5 model for text-to-text tasks
- **t5-base**: Larger T5 model with better quality

## Style Transfer Types

### Formal ↔ Casual
Convert between formal business language and casual conversational style.

**Examples:**
- Formal: "It is with great pleasure that I inform you of our upcoming meeting."
- Casual: "I'm excited to let you know about our upcoming meeting!"

### Sentiment Transfer
Transform negative sentiment to positive while preserving content.

**Examples:**
- Negative: "This product is terrible and doesn't work at all."
- Positive: "This product has room for improvement and could benefit from some enhancements."

### Professional ↔ Friendly
Adjust tone between professional and friendly communication.

**Examples:**
- Professional: "Please submit your report by the end of the week."
- Friendly: "Could you send me your report by Friday?"

## Configuration

The system can be configured via `config/config.yaml`:

```yaml
model_name: "facebook/bart-large-cnn"
max_length: 512
temperature: 0.7
top_p: 0.9
num_beams: 4
early_stopping: true
device: "auto"
batch_size: 1
```

## Evaluation Metrics

- **BLEU Score**: Measures n-gram overlap with reference texts
- **ROUGE Scores**: Measures recall-oriented evaluation metrics
- **Confidence Score**: Model's confidence in the transformation
- **Length Preservation**: Ratio of original to transformed text length

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Or run specific tests:
```bash
python tests/test_style_transfer.py
```

## Advanced Usage

### Custom Model Training

While this project focuses on using pre-trained models, you can extend it for fine-tuning:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Load your custom dataset
# Fine-tune the model
# Save the fine-tuned model
```

### Batch Processing

For large-scale processing:

```python
from src.style_transfer import StyleTransferModel, StyleTransferConfig
from src.data_loader import StyleTransferDataLoader

# Load dataset
loader = StyleTransferDataLoader("data/synthetic_dataset.json")
texts = loader.get_sample_texts(100)

# Process in batches
config = StyleTransferConfig(batch_size=8)
model = StyleTransferModel(config)
results = model.batch_transfer(texts, target_style="casual")
```

## Performance Tips

1. **GPU Acceleration**: The system automatically detects and uses CUDA/MPS if available
2. **Model Selection**: Use smaller models (t5-small, bart-base) for faster inference
3. **Batch Processing**: Process multiple texts together for better efficiency
4. **Caching**: The web interface caches models for faster subsequent loads

## Troubleshooting

### Common Issues

1. **Model Download Errors**: Ensure stable internet connection for first-time model downloads
2. **Memory Issues**: Use smaller models or reduce batch size for limited memory
3. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

### Logging

Check logs for detailed error information:
```bash
tail -f style_transfer.log
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest tests/`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Streamlit for the web interface
- The research community for transformer models and style transfer techniques

## Citation

If you use this project in your research, please cite:

```bibtex
@software{text_style_transfer,
  title={Text Style Transfer: A Modern Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Text-Style-Transfer}
}
```
# Text-Style-Transfer
