"""
Streamlit Web Interface for Text Style Transfer

This module provides a user-friendly web interface for the text style transfer system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from style_transfer import StyleTransferModel, StyleTransferConfig, StyleTransferEvaluator
from data_loader import StyleTransferDataLoader

# Page configuration
st.set_page_config(
    page_title="Text Style Transfer",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name: str):
    """Load and cache the style transfer model."""
    config = StyleTransferConfig(model_name=model_name)
    return StyleTransferModel(config)

@st.cache_data
def load_dataset():
    """Load and cache the dataset."""
    return StyleTransferDataLoader("data/synthetic_dataset.json")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔄 Text Style Transfer</h1>', unsafe_allow_html=True)
    st.markdown("Transform text styles using state-of-the-art transformer models")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "BART Large CNN": "facebook/bart-large-cnn",
            "BART Base": "facebook/bart-base",
            "T5 Small": "t5-small",
            "T5 Base": "t5-base"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        model_name = model_options[selected_model]
        
        # Style transfer options
        st.subheader("🎨 Style Options")
        
        transfer_types = {
            "Formal → Casual": ("formal", "casual"),
            "Casual → Formal": ("casual", "formal"),
            "Negative → Positive": ("negative", "positive"),
            "Professional → Friendly": ("professional", "friendly")
        }
        
        selected_transfer = st.selectbox(
            "Transfer Type",
            options=list(transfer_types.keys()),
            index=0
        )
        
        source_style, target_style = transfer_types[selected_transfer]
        
        # Advanced parameters
        st.subheader("🔧 Advanced Parameters")
        
        max_length = st.slider("Max Length", 50, 512, 256)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        num_beams = st.slider("Number of Beams", 1, 8, 4)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Text", "📊 Batch Processing", "📈 Evaluation", "📚 Dataset Explorer"])
    
    with tab1:
        st.header("Single Text Style Transfer")
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(model_name)
        
        # Text input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_text = st.text_area(
                "Enter text to transform:",
                value="It is with great pleasure that I write to inform you of our upcoming meeting.",
                height=100,
                help="Enter the text you want to transform"
            )
        
        with col2:
            st.markdown("**Quick Examples:**")
            examples = [
                "I would like to express my sincere gratitude for your assistance.",
                "Please be advised that the deadline has been extended.",
                "We regret to inform you that your application has been rejected."
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"Example {i+1}", key=f"example_{i}"):
                    st.session_state.input_text = example
                    st.rerun()
        
        # Process button
        if st.button("🔄 Transform Text", type="primary"):
            if input_text.strip():
                with st.spinner("Processing..."):
                    # Update config with sidebar parameters
                    model.config.max_length = max_length
                    model.config.temperature = temperature
                    model.config.num_beams = num_beams
                    
                    result = model.transfer_style(
                        input_text,
                        target_style=target_style,
                        source_style=source_style
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Original Text")
                        st.markdown(f'<div class="result-box">{result["original_text"]}</div>', 
                                  unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### Transformed Text")
                        st.markdown(f'<div class="result-box">{result["transformed_text"]}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                    
                    with col2:
                        st.metric("Model", selected_model)
                    
                    with col3:
                        st.metric("Transfer Type", selected_transfer)
            else:
                st.warning("Please enter some text to transform.")
    
    with tab2:
        st.header("Batch Processing")
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(model_name)
        
        # Batch input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload CSV file", "Use sample dataset", "Manual input"]
        )
        
        texts_to_process = []
        
        if input_method == "Upload CSV file":
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts_to_process = df['text'].tolist()
                    st.success(f"Loaded {len(texts_to_process)} texts from CSV")
                else:
                    st.error("CSV file must contain a 'text' column")
        
        elif input_method == "Use sample dataset":
            data_loader = load_dataset()
            sample_texts = data_loader.get_sample_texts(10)
            texts_to_process = sample_texts
            st.info(f"Using {len(texts_to_process)} sample texts from dataset")
        
        else:  # Manual input
            manual_texts = st.text_area(
                "Enter texts (one per line):",
                value="It is with great pleasure that I write to inform you of our upcoming meeting.\nI would like to express my sincere gratitude for your assistance.\nPlease be advised that the deadline has been extended.",
                height=150
            )
            texts_to_process = [text.strip() for text in manual_texts.split('\n') if text.strip()]
        
        # Process batch
        if texts_to_process and st.button("🔄 Process Batch", type="primary"):
            with st.spinner("Processing batch..."):
                # Update config
                model.config.max_length = max_length
                model.config.temperature = temperature
                model.config.num_beams = num_beams
                
                results = model.batch_transfer(
                    texts_to_process,
                    target_style=target_style,
                    source_style=source_style
                )
                
                # Display results
                st.subheader("Results")
                
                # Create results DataFrame
                results_df = pd.DataFrame([
                    {
                        "Original": r["original_text"],
                        "Transformed": r["transformed_text"],
                        "Confidence": r.get("confidence", 0)
                    }
                    for r in results
                ])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="style_transfer_results.csv",
                    mime="text/csv"
                )
    
    with tab3:
        st.header("Model Evaluation")
        
        # Load model and dataset
        with st.spinner("Loading model and dataset..."):
            model = load_model(model_name)
            data_loader = load_dataset()
        
        # Evaluation options
        eval_type = st.selectbox(
            "Evaluation Type",
            ["Formal ↔ Casual", "Sentiment Transfer", "Professional ↔ Friendly"]
        )
        
        if eval_type == "Formal ↔ Casual":
            pairs = data_loader.get_formal_casual_pairs()
            source_style, target_style = "formal", "casual"
        elif eval_type == "Sentiment Transfer":
            pairs = data_loader.get_sentiment_pairs()
            source_style, target_style = "negative", "positive"
        else:
            pairs = data_loader.get_professional_friendly_pairs()
            source_style, target_style = "professional", "friendly"
        
        if st.button("📊 Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                # Update config
                model.config.max_length = max_length
                model.config.temperature = temperature
                model.config.num_beams = num_beams
                
                # Process pairs
                results = []
                reference_texts = []
                
                for source, target, category in pairs[:10]:  # Limit for demo
                    result = model.transfer_style(source, target_style, source_style)
                    results.append(result)
                    reference_texts.append(target)
                
                # Evaluate
                evaluator = StyleTransferEvaluator()
                metrics = evaluator.evaluate_transformations(results, reference_texts)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("BLEU Score", f"{metrics.get('bleu', 0):.3f}")
                
                with col2:
                    st.metric("ROUGE-1", f"{metrics.get('rouge1', 0):.3f}")
                
                with col3:
                    st.metric("ROUGE-2", f"{metrics.get('rouge2', 0):.3f}")
                
                with col4:
                    st.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.3f}")
                
                # Visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Confidence Distribution", "Text Length Comparison"),
                    specs=[[{"type": "histogram"}, {"type": "scatter"}]]
                )
                
                # Confidence histogram
                confidences = [r.get("confidence", 0) for r in results]
                fig.add_trace(
                    go.Histogram(x=confidences, name="Confidence"),
                    row=1, col=1
                )
                
                # Length comparison scatter
                original_lengths = [len(r["original_text"]) for r in results]
                transformed_lengths = [len(r["transformed_text"]) for r in results]
                
                fig.add_trace(
                    go.Scatter(
                        x=original_lengths,
                        y=transformed_lengths,
                        mode='markers',
                        name='Length Comparison',
                        text=[f"Confidence: {c:.2f}" for c in confidences]
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Dataset Explorer")
        
        # Load dataset
        data_loader = load_dataset()
        
        # Dataset statistics
        all_pairs = data_loader.get_all_pairs()
        
        st.subheader("Dataset Statistics")
        
        stats_data = []
        for transfer_type, pairs in all_pairs.items():
            stats_data.append({
                "Transfer Type": transfer_type.replace("_", " ").title(),
                "Number of Pairs": len(pairs),
                "Categories": len(set(category for _, _, category in pairs))
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            stats_df,
            x="Transfer Type",
            y="Number of Pairs",
            title="Dataset Distribution by Transfer Type",
            color="Number of Pairs",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample data display
        st.subheader("Sample Data")
        
        selected_transfer_type = st.selectbox(
            "Select transfer type to view:",
            options=list(all_pairs.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        pairs = all_pairs[selected_transfer_type]
        
        if pairs:
            sample_df = pd.DataFrame([
                {
                    "Source": source,
                    "Target": target,
                    "Category": category
                }
                for source, target, category in pairs[:5]
            ])
            
            st.dataframe(sample_df, use_container_width=True)
        
        # Download dataset
        if st.button("📥 Download Full Dataset"):
            dataset_json = data_loader.data
            st.download_button(
                label="Download JSON",
                data=str(dataset_json),
                file_name="style_transfer_dataset.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
