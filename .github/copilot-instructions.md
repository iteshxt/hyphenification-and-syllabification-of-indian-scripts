# Copilot Instructions: Devanagari Syllabification

## Project Overview
Machine learning system for automatic Devanagari word segmentation into syllables (aksharas) using CRF models. Currently in Phase 1 with baseline CRF implementation; Phase 2 plans BiLSTM-CRF comparison.

**Key Goal**: Achieve F1 > 0.90 on syllable boundary detection, extensible to multi-script support.

## Architecture & Data Flow

### Core Pipeline
1. **Raw Data** (`data/devnagri-gold-dataset.jsonl`) → 2. **Format Conversion** (`src/data_converter.py`) → 3. **CRF Format** (`data/crf_train_data_full.txt`) → 4. **Feature Extraction** (`src/features.py`) → 5. **CRF Model** (`src/crf_model.py`) → 6. **Inference** (`src/inference.py`)

### Critical Module Dependencies
- **`src/config.py`**: Single source of truth for ALL paths, thresholds, and script constants (DEVANAGARI_CONSONANTS, DEVANAGARI_VIRAMA, etc.)
  - Never hardcode paths or magic numbers—reference config instead
  - This enables reproducibility and multi-script extension
- **`src/features.py`**: Feature extraction is linguistically complex and tightly coupled to CRF model
  - Features: `syllable` (text), `syllable.length`, `prev_syllable`, `next_syllable`, `has_virama`, `has_vowel_sign`, `starts_with_consonant`, `is_short`, `is_long`
  - Devanagari-specific: virama (्) marks syllable boundaries; vowel signs indicate morphology
- **`src/crf_model.py`**: Wrapper around sklearn's LogisticRegression + DictVectorizer (not true CRF)
  - Design allows future replacement with pycrfsuite or neural models without breaking inference
  - Key methods: `train()`, `predict()`, `predict_proba()`, `save()/load()`
- **`src/inference.py`**: User-facing API (SyllableSegmenter class)
  - Input: word string; Output: list of syllables or with confidence scores
  - Reconstructs syllables by tracking "B" (boundary) predictions

### Data Format
**CRF Training Format** (space-separated lines):
```
syllable TAG
...
(blank line between words)
```
TAG values: `B` (boundary/start) or `O` (inside syllable)

## Coding Patterns & Conventions

### 1. Feature Engineering
- All features are extracted in `src/features.py` using static methods
- Add new features to `FeatureExtractor.extract_syllable_features()` and update docs
- Features must be JSON-serializable (dicts with bool/str/int values only)
- Test new features in `tests/test_features.py` before training

### 2. Configuration Management
- Never use hardcoded paths or constants in model/inference code
- Import from `src/config.py`: `from src.config import CRF_TRAIN_DATA_FULL, DEVANAGARI_VIRAMA, etc.`
- Add script-specific constants to config (e.g., for Hindi support: `HINDI_CONSONANTS`)

### 3. Logging
- All modules use Python's standard `logging` module
- Initialize: `logger = logging.getLogger(__name__)`
- Scripts configure logging with INFO level; models use debug for internal steps
- Check logs for training progress and data loading issues

### 4. Data Validation
- CRF data loading via `DataLoader.load_crf_format()` validates structure
- Synthetic negative examples added via `DataLoader.add_synthetic_negatives()` to prevent class imbalance
- **Never skip validation**—data format errors propagate silently through vectorization

## Development Workflows

### Training Pipeline
```bash
# 1. Preprocess data (if raw JSONL changed)
python scripts/preprocess.py

# 2. Train model (uses crf_train_data_full.txt)
python scripts/train.py
# Output: models/crf_model.pkl + models/metrics.json

# 3. Evaluate metrics
python evaluate_splits.py  # Returns train/test F1, confusion matrix
```

### Inference Workflow
```bash
# Single word
python scripts/infer.py "कर्म"

# Multiple words with confidence
python scripts/infer.py "कर्म" "विद्यालय" --confidence

# Custom model path
python scripts/infer.py "कर्म" --model /path/to/model.pkl
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Specific test file
pytest tests/test_features.py -v
```

## Common Tasks & Decision Points

### Adding a New Feature
1. Define in `FeatureExtractor.extract_syllable_features()` (src/features.py)
2. Add unit test in `tests/test_features.py`
3. Retrain model: `python scripts/train.py`
4. Validate F1 impact using `evaluate_splits.py`

### Multi-Script Extension (Phase 2 Design)
- Create `src/config_hindi.py` (or similar) with Hindi constants
- Update `FeatureExtractor` to accept `language` parameter
- Data loader should be language-agnostic (already is)
- Model can be trained per-language or shared with language embeddings

### Improving Model Performance
- Check feature importance via `model.vectorizer.get_feature_names_out()` + LogisticRegression coefficients
- Adjust `MAX_ITER` in config if training doesn't converge (check logs for "not converged")
- Add more synthetic negatives if class imbalance detected in metrics.json
- Consider train/test split ratio (`TEST_SPLIT_RATIO`) if overfitting

### Switching to True CRF or Neural Model
1. Keep `FeatureExtractor` unchanged—it's model-agnostic
2. Update `src/crf_model.py` class to wrap new model (pycrfsuite, BiLSTM, etc.)
3. Ensure `train()` and `predict()` signatures remain compatible
4. Inference and CLI require zero changes

## Key Files Reference

| File | Purpose | Modify When |
|------|---------|------------|
| `src/config.py` | Project constants | Adding script support, changing thresholds |
| `src/features.py` | Feature engineering | New linguistic features needed |
| `src/crf_model.py` | Model implementation | Switching model type (pycrfsuite, neural) |
| `src/inference.py` | Inference API | Changing output format or adding batch processing |
| `scripts/train.py` | Training entry point | Changing training procedure (e.g., cross-validation) |
| `data/crf_train_data_full.txt` | Training dataset | Adding/removing examples |
| `models/metrics.json` | Evaluation results | Debugging performance regressions |

## Debugging Tips

**Model not converging**: Check MAX_ITER in config; increase if warning appears in logs
**Low F1 scores**: Validate training data format (blank lines between words); check feature extraction with test_features.py
**Inference crashes**: Ensure model file exists; check word is valid Devanagari (not ASCII)
**Features seem redundant**: Check vectorizer output dimensionality in metrics.json
