import streamlit as st
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from pathlib import Path
import json
import time
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .sentiment-positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================
st.title("🎭 Sentiment Analysis Dashboard")
st.markdown("### AI-Powered Product Review Analysis using DistilBERT")
st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Confidence threshold
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for positive/negative classification"
    )
    
    st.markdown("---")
    
    st.header("📊 Model Information")
    st.info("""
    **Model:** DistilBERT (Fine-tuned)
    
    **Dataset:** Amazon Product Reviews
    
    **Performance:**
    - TextBlob Baseline: 68%
    - DistilBERT: 90%+
    - Improvement: +22%+
    
    **Classes:**
    - 😊 Positive
    - 😞 Negative
    """)
    
    st.markdown("---")
    
    st.header("📖 How to Use")
    st.markdown("""
    1. **Single Review:** Analyze one review
    2. **Batch Analysis:** Upload CSV file
    3. **Aspect Analysis:** Analyze specific aspects
    4. **Demo Reviews:** Try sample reviews
    """)
    
    st.markdown("---")
    st.markdown("**Built with:** DistilBERT & Streamlit")

# ============================================================
# LOAD MODEL
# ============================================================
import os
import gdown
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
import streamlit as st

# ============================================================
# GOOGLE DRIVE FILE IDs
# Replace these with your actual Google Drive file IDs
# ============================================================
GDRIVE_FILES = {
    'model': '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74',        # best_model.pt
    'tokenizer_config': '1abc123...',                         # config.json
    'tokenizer_json': '1def456...',                           # tokenizer.json
    'tokenizer_config2': '1ghi789...',                        # tokenizer_config.json
    'vocab': '1jkl012...',                                    # vocab.txt
}

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, output_path, quiet=False)

    return os.path.exists(output_path)


@st.cache_resource
def load_model_and_tokenizer():
    try:
        model_path = 'models/best_model.pt'
        tokenizer_path = 'models/tokenizer'

        # Download model if not exists
        if not os.path.exists(model_path):
            st.info("Downloading model... please wait (one time only)")

            success = download_from_gdrive(
                GDRIVE_FILES['model'],
                model_path
            )

            if not success:
                return None, None, "Failed to download model"

        # Download tokenizer files if not exists
        tokenizer_files = {
            'config.json': GDRIVE_FILES['tokenizer_config'],
            'tokenizer.json': GDRIVE_FILES['tokenizer_json'],
            'tokenizer_config.json': GDRIVE_FILES['tokenizer_config2'],
            'vocab.txt': GDRIVE_FILES['vocab']
        }

        for filename, file_id in tokenizer_files.items():
            filepath = os.path.join(tokenizer_path, filename)

            if not os.path.exists(filepath):
                download_from_gdrive(file_id, filepath)

        # Load device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

        # Load model
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        return model, tokenizer, None

    except Exception as e:
        return None, None, str(e)

with st.spinner('🔄 Loading AI model...'):
    model, tokenizer, error = load_model_and_tokenizer()

if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("""
    **Model files not found!**
    
    Make sure these files exist:
    - `models/best_model.pt`
    - `models/tokenizer/`
    
    Run the training notebook first to generate these files.
    """)
    st.stop()
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.success("✅ Model loaded successfully!")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def preprocess_text(text):
    """Clean text"""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_sentiment(text, model, tokenizer, device, threshold=0.5, max_length=256):
    """Predict sentiment for a single text"""
    
    encoding = tokenizer(
        str(text),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    positive_prob = probs[0][1].item()
    negative_prob = probs[0][0].item()
    
    if positive_prob >= threshold:
        sentiment = 'POSITIVE'
        confidence = positive_prob
    else:
        sentiment = 'NEGATIVE'
        confidence = negative_prob
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'positive_prob': positive_prob,
        'negative_prob': negative_prob
    }


# Aspect keywords
ASPECTS = {
    'Quality': ['quality', 'build', 'material', 'durable', 'sturdy', 'cheap', 'flimsy', 'solid'],
    'Price': ['price', 'cost', 'value', 'expensive', 'cheap', 'worth', 'affordable', 'overpriced'],
    'Shipping': ['shipping', 'delivery', 'arrived', 'package', 'fast', 'slow', 'delayed', 'transit'],
    'Service': ['service', 'support', 'help', 'staff', 'response', 'customer', 'refund', 'return'],
    'Performance': ['performance', 'works', 'function', 'speed', 'efficient', 'effective', 'powerful'],
    'Design': ['design', 'look', 'appearance', 'style', 'color', 'size', 'weight', 'compact']
}


def extract_aspect_sentiments(text, model, tokenizer, device, threshold=0.5):
    """Extract sentiment for each aspect"""
    
    text_lower = text.lower()
    results = {}
    
    for aspect, keywords in ASPECTS.items():
        mentioned = any(kw in text_lower for kw in keywords)
        
        if mentioned:
            sentences = text.split('.')
            aspect_sentences = [
                s.strip() for s in sentences
                if any(kw in s.lower() for kw in keywords)
            ]
            
            if aspect_sentences:
                combined = ' '.join(aspect_sentences[:3])
                result = predict_sentiment(combined, model, tokenizer, device, threshold)
                results[aspect] = {
                    'mentioned': True,
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'positive_prob': result['positive_prob'],
                    'negative_prob': result['negative_prob']
                }
        else:
            results[aspect] = {'mentioned': False}
    
    return results


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "✍️ Single Review",
    "📁 Batch Analysis",
    "🔍 Aspect Analysis",
    "🎯 Demo Reviews"
])

# ============================================================
# TAB 1: SINGLE REVIEW
# ============================================================
with tab1:
    st.header("✍️ Analyze Single Review")
    
    review_text = st.text_area(
        "Enter your product review:",
        height=150,
        placeholder="Type your review here... e.g., 'This product is amazing! Great quality and fast shipping.'"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        review_text = ""
        if 'single_result' in st.session_state:
            del st.session_state.single_result
    
    if analyze_btn and review_text:
        with st.spinner("🤔 Analyzing sentiment..."):
            start_time = time.time()
            result = predict_sentiment(
                review_text, model, tokenizer, device, threshold
            )
            inference_time = time.time() - start_time
            st.session_state.single_result = result
        
    if 'single_result' in st.session_state and review_text:
        result = st.session_state.single_result
        
        st.markdown("---")
        
        # Result display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Sentiment Result")
            
            if result['sentiment'] == 'POSITIVE':
                st.markdown("""
                <div class='sentiment-positive'>
                    <h2>😊 POSITIVE</h2>
                    <p>This review expresses positive sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='sentiment-negative'>
                    <h2>😞 NEGATIVE</h2>
                    <p>This review expresses negative sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            with m2:
                st.metric("😊 Positive", f"{result['positive_prob']*100:.1f}%")
            with m3:
                st.metric("😞 Negative", f"{result['negative_prob']*100:.1f}%")
        
        with col2:
            st.subheader("📈 Probability Distribution")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['positive_prob'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Positive Sentiment Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ff4444"},
                        {'range': [40, 60], 'color': "#ffaa00"},
                        {'range': [60, 100], 'color': "#44bb44"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Review statistics
        st.markdown("---")
        st.subheader("📋 Review Statistics")
        
        words = review_text.split()
        sentences = review_text.split('.')
        
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Word Count", len(words))
        with s2:
            st.metric("Sentences", len([s for s in sentences if s.strip()]))
        with s3:
            st.metric("Characters", len(review_text))
        with s4:
            st.metric("Avg Word Length", f"{np.mean([len(w) for w in words]):.1f}")
    
    elif analyze_btn and not review_text:
        st.warning("⚠️ Please enter a review to analyze!")

# ============================================================
# TAB 2: BATCH ANALYSIS
# ============================================================
with tab2:
    st.header("📁 Batch Analysis")
    st.markdown("Upload a CSV file with reviews to analyze multiple reviews at once.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV must have a column named 'review' or 'text' or 'content'"
    )
    
    # Sample CSV template
    with st.expander("📥 Download Sample CSV Template"):
        sample_data = pd.DataFrame({
            'review': [
                'This product is absolutely amazing! Best purchase ever!',
                'Terrible quality. Broke after 2 days. Very disappointed.',
                'Good value for money. Does what it says.',
                'Fast shipping, great packaging. Love it!',
                'Not worth the price. Expected much better quality.'
            ]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name='review_template.csv',
            mime='text/csv'
        )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Find text column
        text_col = None
        for col in ['review', 'text', 'content', 'Review', 'Text', 'Content']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            st.error("❌ Could not find text column! Please name it 'review', 'text', or 'content'")
        else:
            st.success(f"✅ Found {len(df)} reviews in column: '{text_col}'")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Analyze All Reviews", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for i, (idx, row) in enumerate(df.iterrows()):
                    text = str(row[text_col])
                    result = predict_sentiment(text, model, tokenizer, device, threshold)
                    results.append({
                        'review': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': result['sentiment'],
                        'confidence': f"{result['confidence']*100:.1f}%",
                        'positive_prob': f"{result['positive_prob']*100:.1f}%",
                        'negative_prob': f"{result['negative_prob']*100:.1f}%"
                    })
                    
                    progress = (i + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing review {i+1}/{len(df)}...")
                
                status_text.text("✅ Analysis complete!")
                results_df = pd.DataFrame(results)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Summary metrics
                pos_count = (results_df['sentiment'] == 'POSITIVE').sum()
                neg_count = (results_df['sentiment'] == 'NEGATIVE').sum()
                pos_pct = pos_count / len(results_df) * 100
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Reviews", len(results_df))
                with m2:
                    st.metric("😊 Positive", f"{pos_count} ({pos_pct:.1f}%)")
                with m3:
                    st.metric("😞 Negative", f"{neg_count} ({100-pos_pct:.1f}%)")
                with m4:
                    st.metric("Positive Rate", f"{pos_pct:.1f}%")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig = px.pie(
                        values=[pos_count, neg_count],
                        names=['Positive', 'Negative'],
                        color_discrete_sequence=['#2ecc71', '#e74c3c'],
                        title='Sentiment Distribution'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Bar chart by confidence
                    conf_data = results_df.copy()
                    conf_data['conf_num'] = conf_data['confidence'].str.replace('%', '').astype(float)
                    
                    fig = px.histogram(
                        conf_data,
                        x='conf_num',
                        color='sentiment',
                        color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'},
                        title='Confidence Distribution',
                        labels={'conf_num': 'Confidence (%)', 'count': 'Count'},
                        barmode='overlay',
                        opacity=0.7
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("#### 📋 Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name='sentiment_results.csv',
                    mime='text/csv'
                )

# ============================================================
# TAB 3: ASPECT ANALYSIS
# ============================================================
with tab3:
    st.header("🔍 Aspect-Based Sentiment Analysis")
    st.markdown("Analyze sentiment for specific aspects of a product review.")
    
    aspect_text = st.text_area(
        "Enter a detailed product review:",
        height=200,
        placeholder="""Example: The quality of this product is excellent and very durable.
The price is a bit high but worth it for the performance.
Shipping was fast and delivery was on time.
Customer service was very helpful."""
    )
    
    if st.button("🔍 Analyze Aspects", type="primary") and aspect_text:
        with st.spinner("🔍 Analyzing aspects..."):
            # Overall sentiment
            overall = predict_sentiment(
                aspect_text, model, tokenizer, device, threshold
            )
            
            # Aspect sentiments
            aspects = extract_aspect_sentiments(
                aspect_text, model, tokenizer, device, threshold
            )
        
        # Overall result
        st.markdown("---")
        
        overall_col1, overall_col2 = st.columns([1, 2])
        
        with overall_col1:
            st.subheader("Overall Sentiment")
            
            if overall['sentiment'] == 'POSITIVE':
                st.markdown("""
                <div class='sentiment-positive'>
                    <h3>😊 POSITIVE</h3>
                    <p>Overall positive review</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='sentiment-negative'>
                    <h3>😞 NEGATIVE</h3>
                    <p>Overall negative review</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.metric(
                "Confidence",
                f"{overall['confidence']*100:.1f}%"
            )
        
        with overall_col2:
            st.subheader("Aspect Breakdown")
            
            # Filter mentioned aspects
            mentioned = {k: v for k, v in aspects.items() if v.get('mentioned', False)}
            not_mentioned = [k for k, v in aspects.items() if not v.get('mentioned', False)]
            
            if mentioned:
                # Create aspect cards
                aspect_cols = st.columns(min(3, len(mentioned)))
                
                for i, (aspect, data) in enumerate(mentioned.items()):
                    with aspect_cols[i % 3]:
                        color = "#d4edda" if data['sentiment'] == 'POSITIVE' else "#f8d7da"
                        emoji = "✅" if data['sentiment'] == 'POSITIVE' else "❌"
                        border = "#28a745" if data['sentiment'] == 'POSITIVE' else "#dc3545"
                        
                        st.markdown(f"""
                        <div style='background-color:{color}; border:2px solid {border};
                                    border-radius:8px; padding:10px; margin:5px; text-align:center;'>
                            <strong>{emoji} {aspect}</strong><br>
                            <span>{data['sentiment']}</span><br>
                            <small>Confidence: {data['confidence']*100:.1f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No specific aspects detected in this review.")
            
            if not_mentioned:
                st.caption(f"⚪ Not mentioned: {', '.join(not_mentioned)}")
        
        # Radar chart for aspects
        if mentioned:
            st.markdown("---")
            st.subheader("📊 Aspect Sentiment Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart
                categories = list(mentioned.keys())
                values = [v['positive_prob'] * 100 for v in mentioned.values()]
                values.append(values[0])  # Close the polygon
                categories_closed = categories + [categories[0]]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories_closed,
                    fill='toself',
                    line=dict(color='#3498db', width=2),
                    fillcolor='rgba(52, 152, 219, 0.3)',
                    name='Positive Score'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    title='Aspect Sentiment Radar',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Horizontal bar chart
                aspect_df = pd.DataFrame({
                    'Aspect': list(mentioned.keys()),
                    'Positive Score': [v['positive_prob'] * 100 for v in mentioned.values()],
                    'Sentiment': [v['sentiment'] for v in mentioned.values()]
                })
                
                fig = px.bar(
                    aspect_df,
                    x='Positive Score',
                    y='Aspect',
                    color='Sentiment',
                    color_discrete_map={
                        'POSITIVE': '#2ecc71',
                        'NEGATIVE': '#e74c3c'
                    },
                    title='Aspect Sentiment Scores',
                    orientation='h',
                    range_x=[0, 100]
                )
                
                fig.add_vline(
                    x=50,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Neutral"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    elif st.button("🔍 Analyze Aspects", key="empty_check"):
        st.warning("⚠️ Please enter a review to analyze!")

# ============================================================
# TAB 4: DEMO REVIEWS
# ============================================================
with tab4:
    st.header("🎯 Demo Reviews")
    st.markdown("Try these sample reviews to see the model in action!")
    
    # Sample reviews
    demo_reviews = {
        "⭐⭐⭐⭐⭐ - Excellent Product": """
        This product exceeded all my expectations! The build quality is absolutely fantastic,
        very durable and sturdy. The price is reasonable for the quality you get.
        Shipping was incredibly fast - arrived in just 2 days!
        Customer service was also very helpful when I had a question.
        Highly recommend to everyone!
        """,
        
        "⭐ - Very Disappointed": """
        Extremely disappointed with this purchase. The quality is terrible - 
        it broke after just one week of normal use. The materials feel very cheap and flimsy.
        Way too expensive for what you actually get.
        Shipping took 3 weeks and the package was damaged.
        Customer service was completely unhelpful and rude.
        Would not recommend to anyone.
        """,
        
        "⭐⭐⭐ - Mixed Experience": """
        The product design looks great and the performance is decent.
        However, the price is quite high for the quality of materials used.
        Shipping was fast but the customer service could be better.
        Overall it works as described but I expected more.
        """,
        
        "⭐⭐⭐⭐ - Great Value": """
        Really happy with this purchase! Great value for money.
        The quality is solid and it works exactly as described.
        Fast delivery and well packaged.
        Minor issue with one feature but customer service helped resolve it quickly.
        Would definitely buy again!
        """,
        
        "⭐⭐ - Not Worth It": """
        Save your money and look elsewhere. The performance is poor 
        and it stopped working after just a month.
        The design is okay but the build quality is disappointing.
        Customer support never responded to my emails.
        Very overpriced for such a low quality item.
        """
    }
    
    selected_demo = st.selectbox(
        "Select a demo review:",
        list(demo_reviews.keys())
    )
    
    demo_text = demo_reviews[selected_demo]
    
    st.markdown("**Selected Review:**")
    st.text_area("Review text:", value=demo_text, height=150, disabled=True)
    
    if st.button("🚀 Analyze Demo Review", type="primary"):
        with st.spinner("🤔 Analyzing..."):
            # Overall sentiment
            overall = predict_sentiment(
                demo_text, model, tokenizer, device, threshold
            )
            
            # Aspect sentiments
            aspects = extract_aspect_sentiments(
                demo_text, model, tokenizer, device, threshold
            )
        
        st.markdown("---")
        
        # Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Sentiment")
            
            if overall['sentiment'] == 'POSITIVE':
                st.success(f"### 😊 POSITIVE ({overall['confidence']*100:.1f}%)")
            else:
                st.error(f"### 😞 NEGATIVE ({overall['confidence']*100:.1f}%)")
            
            # Probabilities
            prob_df = pd.DataFrame({
                'Sentiment': ['😊 Positive', '😞 Negative'],
                'Probability': [
                    overall['positive_prob'] * 100,
                    overall['negative_prob'] * 100
                ]
            })
            
            fig = px.bar(
                prob_df,
                x='Sentiment',
                y='Probability',
                color='Sentiment',
                color_discrete_map={
                    '😊 Positive': '#2ecc71',
                    '😞 Negative': '#e74c3c'
                },
                title='Sentiment Probabilities',
                range_y=[0, 100]
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Aspect Analysis")
            
            mentioned = {k: v for k, v in aspects.items() if v.get('mentioned', False)}
            
            if mentioned:
                aspect_data = []
                for aspect, data in mentioned.items():
                    aspect_data.append({
                        'Aspect': aspect,
                        'Sentiment': data['sentiment'],
                        'Score': f"{data['positive_prob']*100:.1f}%",
                        'Status': '✅' if data['sentiment'] == 'POSITIVE' else '❌'
                    })
                
                aspect_df = pd.DataFrame(aspect_data)
                st.dataframe(aspect_df, use_container_width=True)
                
                # Aspect chart
                fig = px.bar(
                    aspect_df,
                    x='Aspect',
                    y=[float(s.replace('%', '')) for s in aspect_df['Score']],
                    color='Sentiment',
                    color_discrete_map={
                        'POSITIVE': '#2ecc71',
                        'NEGATIVE': '#e74c3c'
                    },
                    title='Aspect Scores',
                    labels={'y': 'Positive Score (%)'}
                )
                fig.add_hline(
                    y=50, line_dash="dash",
                    line_color="gray",
                    annotation_text="Threshold"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No specific aspects detected.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Sentiment Analysis Dashboard</strong> | 
    Built with DistilBERT & Streamlit</p>
    <p style='font-size: 12px;'>
    🎓 NLP Project | Fine-tuned on Amazon Reviews | 
    Accuracy: 90%+ | Improvement over baseline: +22%
    </p>
</div>
""", unsafe_allow_html=True)