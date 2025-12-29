# Devanagari Syllabification

**Automatic syllable segmentation for Devanagari (Hindi) using BiLSTM+CRF & CRF models with a modern web interface.**

## Overview

ML system for segmenting Devanagari words into syllables with a React frontend and FastAPI backend.

- Dual Models: BiLSTM+CRF (neural) & CRF (traditional)
- High Accuracy: F1 > 0.90 on both models
- Fast Inference: Real-time segmentation
- Web Interface: Modern neobrutalist UI with transliteration support
- Transliteration: Type in English (romanized) → auto-converts to Devanagari

## Project Structure

```
hyphenification-and-syllabification-of-indian-scripts/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── App.css          # Neobrutalist styling
│   │   └── index.css        # Global styles
│   ├── index.html
│   └── package.json
│
├── server/                   # Python backend
│   ├── app.py               # FastAPI server
│   ├── src/
│   │   ├── config.py        # Configuration & paths
│   │   ├── features.py      # Feature extraction
│   │   ├── crf_model.py     # CRF model
│   │   ├── bilstm_crf_model.py  # BiLSTM+CRF model
│   │   ├── inference.py     # Unified inference API
│   │   └── syllable_splitter.py
│   ├── scripts/
│   │   ├── train.py         # Train CRF
│   │   └── train_bilstm_crf.py  # Train BiLSTM+CRF
│   ├── data/
│   │   ├── devnagri-gold-dataset.jsonl
│   │   └── crf_train_data_full.txt
│   ├── models/
│   │   ├── crf_model.pkl
│   │   └── bilstm_crf_model.pkl
│   ├── tests/
│   ├── main.py              # CLI interface
│   └── requirements.txt
│
└── README.md
```

## Quick Start

### 1. Setup Backend

```bash
cd server

# Using uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Backend Server

```bash
cd server
uv run uvicorn app:app --reload --port 8000

# Server runs at http://localhost:8000
```

### 3. Setup & Run Frontend

```bash
cd frontend
npm install
npm run dev

# Frontend runs at http://localhost:5173
```

### 4. Open in Browser

Visit http://localhost:5173 to use the web interface.

## Web Interface Features

- **Transliteration**: Type in English (e.g., "bharat") and it auto-converts to Devanagari (भारत)
- **Live Preview**: See the Devanagari conversion as you type
- **Dual Input**: Works with both romanized English and direct Devanagari input
- **Example Buttons**: Quick examples to try (bharat, namaste, vidyalaya, etc.)
- **Syllable Visualization**: See syllables as separate chips
- **Hyphenated Output**: Shows word with syllable boundaries marked
- **Statistics**: Syllable count, character count, input type

## API Endpoints

### POST /segment

Segment a single word into syllables.

```bash
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{"word": "भारत"}'
```

Response:
```json
{
  "word": "भारत",
  "syllables": ["भा", "रत"],
  "hyphenated": "भा-रत",
  "count": 2
}
```

### POST /segment/batch

Segment multiple words.

```bash
curl -X POST http://localhost:8000/segment/batch \
  -H "Content-Type: application/json" \
  -d '["भारत", "नमस्ते"]'
```

## CLI Usage

```bash
cd server
python main.py              # BiLSTM+CRF (default)
python main.py --crf        # CRF model

# Enter: कर्म → Output: ['कर्', 'म']
```

## Python API

```python
from src.inference import SyllableSegmenter
from src.config import BILSTM_CRF_MODEL_PATH

segmenter = SyllableSegmenter(BILSTM_CRF_MODEL_PATH)
print(segmenter.segment_word('कर्म'))       # ['कर्', 'म']
print(segmenter.segment_word('विद्यालय'))  # ['वि', 'द्या', 'लय']
```

## Models

### BiLSTM+CRF (Recommended)

| Metric | Score |
|--------|-------|
| Precision | 1.0 |
| Recall | 1.0 |
| F1-Score | 1.0 |
| Accuracy | 1.0 |

- **Architecture**: Embedding (64D) → BiLSTM (128H) → CRF
- **Training**: 645 sentences, 50 epochs
- **Model Size**: 2.5MB | **Inference**: 5-10ms/word
- **File**: `server/models/bilstm_crf_model.pkl`

### CRF Baseline

| Metric | Score |
|--------|-------|
| Precision | 0.92 |
| Recall | 0.89 |
| F1-Score | 0.91 |
| Accuracy | 0.90 |

- **Architecture**: DictVectorizer + LogisticRegression
- **Training**: 516 train, 129 test
- **Model Size**: 200KB | **Inference**: 1-2ms/word
- **File**: `server/models/crf_model.pkl`

## Training

```bash
cd server

# Preprocess (JSONL → CRF format)
python scripts/preprocess.py

# Train CRF
python scripts/train.py

# Train BiLSTM+CRF
python scripts/train_bilstm_crf.py
```

## Tech Stack

- **Frontend**: React, Vite, Vanilla CSS (Neobrutalist design)
- **Backend**: FastAPI, Uvicorn
- **ML**: PyTorch, scikit-learn
- **Models**: BiLSTM+CRF, CRF

## Transliteration Examples

| Input (English) | Output (Devanagari) |
|-----------------|---------------------|
| bharat | भारत |
| namaste | नमस्ते |
| vidyalaya | विद्यालय |
| shiksha | शिक्षा |

## Common Issues

**Model not found?** Train: `python scripts/train_bilstm_crf.py`

**Backend not connecting?** Ensure server is running on port 8000

**CORS errors?** Backend includes CORS middleware for localhost

**Low accuracy on new words?** Use BiLSTM+CRF (better for unseen words)

## References

- CRF: https://en.wikipedia.org/wiki/Conditional_random_field
- BiLSTM: https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks
- Devanagari: https://en.wikipedia.org/wiki/Devanagari
