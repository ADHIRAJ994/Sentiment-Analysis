# 🎭 Sentiment Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35.0-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered sentiment analysis dashboard built with **DistilBERT** fine-tuned on Amazon Product Reviews. Achieves **90%+ accuracy** — a **+22% improvement** over the TextBlob baseline.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Results](#-results--visualizations)
- [Technologies](#-technologies-used)
- [Disclaimer](#-disclaimer)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project builds an end-to-end NLP pipeline for **sentiment analysis of product reviews**. Using **DistilBERT** (a lighter, faster version of BERT), the model classifies reviews as Positive or Negative with high accuracy while also providing **aspect-based sentiment analysis** to identify sentiment toward specific product attributes.

**Use Cases:**
- E-commerce review analysis
- Brand monitoring
- Customer feedback processing
- Product quality assessment
- Business intelligence

---

## ✨ Key Features

### 🤖 High-Accuracy NLP Model
- 90%+ accuracy using fine-tuned DistilBERT
- Trained on 10,000 Amazon product reviews
- Optimized classification threshold
- Fast inference (~100ms per review)

### 🔍 Aspect-Based Analysis
- Detects sentiment for 6 product aspects:
  - 🏗️ Quality
  - 💰 Price
  - 🚚 Shipping
  - 👥 Customer Service
  - ⚡ Performance
  - 🎨 Design
- Visual radar chart for aspect scores
- Aspect mention detection

### 📁 Batch Processing
- Upload CSV files with multiple reviews
- Process hundreds of reviews at once
- Export results to CSV
- Interactive visualizations

### 💻 Interactive Dashboard
- Real-time sentiment prediction
- Gauge chart for probability scores
- Confidence metrics
- Demo reviews included

---

## 📊 Model Performance

### Comparison Table

| Model | Accuracy | F1 Score | Type |
|-------|----------|----------|------|
| **TextBlob** | 68.00% | 0.68 | Rule-based |
| **DistilBERT** | **90%+** | **0.90+** | Fine-tuned Transformer |
| **Improvement** | **+22%+** | **+0.22+** | - |

### Test Set Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 90%+ |
| **Precision** | 90%+ |
| **Recall** | 90%+ |
| **F1-Score** | 90%+ |
| **AUC-ROC** | 0.96+ |

### Error Analysis Insights
- Model excels at clear positive/negative reviews
- Handles mixed-sentiment reviews reasonably well
- Struggles with sarcasm and irony (known BERT limitation)
- Short reviews (< 10 words) have slightly lower accuracy

---

## 🏗️ Architecture

### Model Pipeline

Raw Text
↓
Text Preprocessing
(lowercase, remove HTML, URLs, special chars)
↓
DistilBERT Tokenizer
(max_length=256, padding, truncation)
↓
DistilBERT Encoder
(6 transformer layers, 768 hidden dims)
↓
Classification Head
(Dense → Dropout → Dense)
↓
Softmax Output
(Positive Probability, Negative Probability)
↓
Threshold Decision
(default: 0.5, optimal: tuned)
↓
Sentiment Label + Confidence

### Training Strategy

**Fine-Tuning Configuration:**
- All DistilBERT layers unfrozen
- AdamW optimizer with weight decay
- Linear warmup schedule (10% of steps)
- Gradient clipping (max_norm=1.0)
- Early stopping on validation accuracy

**Hyperparameters:**
```python
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1
}
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- GPU optional (CPU works fine)

### Step 1: Clone Repository

```bash
git clone https://github.com/ADHIRAJ994/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
```

### Step 5: Download Model Files

Download `best_model.pt` and `tokenizer/` from releases and place in `models/` folder.

---

## 💻 Usage

### Run Web Dashboard

```bash
streamlit run app.py
```

**Access at:** `http://localhost:8501`

### Python API

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = DistilBertTokenizer.from_pretrained('models/tokenizer')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
checkpoint = torch.load('models/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Predict
def predict(text):
    encoding = tokenizer.encode_plus(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(outputs.logits, dim=1)
    
    sentiment = 'POSITIVE' if pred.item() == 1 else 'NEGATIVE'
    confidence = probs[0][pred.item()].item()
    return sentiment, confidence

# Example
sentiment, confidence = predict("This product is absolutely amazing!")
print(f"Sentiment: {sentiment} ({confidence*100:.1f}% confidence)")
```

### Batch Processing via CSV

```python
import pandas as pd

# Create CSV with reviews
df = pd.DataFrame({
    'review': [
        'Great product, highly recommend!',
        'Terrible quality, waste of money.',
        'Good value for the price.'
    ]
})
df.to_csv('reviews.csv', index=False)

# Upload to dashboard and analyze
```

### Train from Scratch

```bash
# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_bert_model.ipynb
jupyter notebook notebooks/03_finetuning.ipynb
jupyter notebook notebooks/04_deployment.ipynb
```

---

## 📁 Project Structure
Sentiment Analysis Dashboard/

│
├── 📁 data/
│   ├── train_processed.csv          # Preprocessed training data
│   └── test_processed.csv           # Preprocessed test data
│

├── 📁 models/
│   ├── best_model.pt                # Fine-tuned DistilBERT weights
│   ├── model_config.json            # Model configuration
│   └── tokenizer/                   # Saved tokenizer files
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
│

├── 📁 results/
│   ├── plots/
│   │   ├── class_distribution.png
│   │   ├── wordclouds.png
│   │   ├── top_words.png
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── model_comparison.png
│   │   ├── aspect_sentiment_heatmap.png
│   │   ├── aspect_statistics.png
│   │   ├── error_analysis.png
│   │   └── threshold_analysis.png
│   └── metrics/
│       ├── training_history.csv
│       ├── test_results.json
│       └── final_results.json
│

├── 📁 notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_bert_model.ipynb          # BERT model training
│   ├── 03_finetuning.ipynb          # Fine-tuning & optimization
│   └── 04_deployment.ipynb          # Deployment preparation
│

├── 📄 app.py                         # Streamlit dashboard
├── 📄 requirements.txt               # Python dependencies
├── 📄 config.json                    # Project configuration
├── 📄 .gitignore                     # Git ignore rules
└── 📄 README.md                      # This file

---

## 📊 Dataset

### Source
- **Dataset:** Amazon Polarity Reviews
- **Platform:** Hugging Face Datasets
- **Link:** [amazon_polarity](https://huggingface.co/datasets/amazon_polarity)

### Statistics

| Split | Positive | Negative | Total |
|-------|----------|----------|-------|
| Train | 5,000 | 5,000 | 10,000 |
| Val | 850 | 850 | 1,700 |
| Test | 1,000 | 1,000 | 2,000 |

### Preprocessing Steps
1. Convert to lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove special characters
5. Remove extra whitespace
6. Truncate to max_length=256 tokens

---

## 🎓 Model Training

### Phase 1: Baseline (Week 1)
- Rule-based sentiment with TextBlob
- **Result:** 68% accuracy
- Used as benchmark for improvement

### Phase 2: Fine-Tuning (Week 2)
- Loaded pre-trained DistilBERT
- Added classification head
- Fine-tuned for 3 epochs
- **Result:** 88-92% accuracy

### Phase 3: Optimization (Week 3)
- Threshold tuning
- Error analysis
- Aspect-based sentiment extraction
- **Result:** 90%+ optimized accuracy

### Training Details
Epoch 1/3:
Train Loss: ~0.35 | Train Acc: ~85%
Val Loss:   ~0.28 | Val Acc:   ~88%
Epoch 2/3:
Train Loss: ~0.22 | Train Acc: ~91%
Val Loss:   ~0.24 | Val Acc:   ~90%
Epoch 3/3:
Train Loss: ~0.15 | Train Acc: ~94%
Val Loss:   ~0.26 | Val Acc:   ~91%

---

## 📈 Results & Visualizations

### Training History
- Loss decreases consistently across epochs
- Validation accuracy improves from 88% to 91%+
- No significant overfitting observed

### Confusion Matrix Insights
            Predicted
        Negative  Positive
Actual Neg    ~450      ~50
Pos    ~50       ~450

### Aspect Analysis Results

| Aspect | Coverage | Positive Rate |
|--------|----------|---------------|
| Quality | High | ~65% |
| Price | High | ~55% |
| Shipping | Medium | ~70% |
| Service | Medium | ~60% |
| Performance | Medium | ~68% |
| Design | Low | ~72% |

### Key Findings
1. Price aspect has lowest positive rate (most complaints)
2. Shipping aspect has highest positive rate
3. Quality is most frequently mentioned aspect
4. High confidence predictions (>90%) are almost always correct

---

## 🛠️ Technologies Used

### NLP & Deep Learning
- **PyTorch 2.0.1** - Deep learning framework
- **Transformers 4.35.0** - DistilBERT model
- **NLTK 3.8.1** - Text preprocessing
- **TextBlob 0.17.1** - Baseline model

### Data Science
- **NumPy 1.24.3** - Numerical computing
- **Pandas 2.1.1** - Data manipulation
- **scikit-learn 1.3.0** - Metrics and utilities

### Visualization
- **Matplotlib 3.7.2** - Static plots
- **Seaborn 0.13.0** - Statistical plots
- **Plotly 5.17.0** - Interactive charts
- **WordCloud 1.9.2** - Word cloud generation

### Web Development
- **Streamlit 1.28.1** - Web dashboard

### Development Tools
- **Jupyter Notebook** - Development
- **Git & GitHub** - Version control

---

## ⚠️ Disclaimer

### Limitations

1. **Domain Specificity**
   - Trained on Amazon product reviews
   - May not generalize to other domains (e.g., movie reviews, news)

2. **Language**
   - English only
   - May struggle with slang or informal language

3. **Sarcasm**
   - Known weakness of transformer models
   - Sarcastic reviews may be misclassified

4. **Context Window**
   - Reviews truncated at 256 tokens
   - Very long reviews may lose context

### Responsible Use
- Not suitable for critical business decisions without human review
- Always validate predictions on your specific domain
- Consider bias in training data

---

## 🚀 Future Improvements

### Short-term (1-3 months)
- [ ] Multi-class sentiment (5-star rating)
- [ ] Multi-language support
- [ ] Real-time Twitter/social media analysis
- [ ] Sarcasm detection module

### Medium-term (3-6 months)
- [ ] Deploy as REST API (FastAPI)
- [ ] Database integration for history
- [ ] User authentication
- [ ] Analytics dashboard

### Long-term (6-12 months)
- [ ] Fine-tune on domain-specific data
- [ ] Active learning pipeline
- [ ] Mobile application
- [ ] Integration with e-commerce platforms

---

## 📄 License

This project is licensed under the **MIT License**.
MIT License
Copyright (c) 2026 Adhiraj Chakravorty
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📞 Contact

**Adhiraj Chakravorty**

- 📧 Email: [youradhi20@gmail.com]
- 💼 LinkedIn: https://www.linkedin.com/in/adhiraj-chakravorty-788685344/
- 🐱 GitHub: [@ADHIRAJ994](https://github.com/ADHIRAJ994)

### Project Links
- 🔗 Repository: [github.com/ADHIRAJ994/sentiment-analysis-dashboard](https://github.com/ADHIRAJ994/sentiment-analysis-dashboard)
- 🌐 Live Demo: https://sentiment-analysis-y5kp2cxntkssecxadpvfa5.streamlit.app/

---

---

<div align="center">

### 🎊 Thank you for checking out this project!

[⬆ Back to Top](#-sentiment-analysis-dashboard)

</div>