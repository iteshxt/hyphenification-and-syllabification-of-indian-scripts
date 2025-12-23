# Copilot Instructions: Devanagari Syllabification

## Project Overview
Machine learning system for automatic Devanagari word segmentation into syllables (aksharas). 

**Status**: Phase 1 (CRF baseline ✅ complete, F1 > 0.90) + Phase 2 (BiLSTM+CRF 🔄 in progress)

**Key Goal**: Maintain F1 > 0.90 on syllable boundary detection while enabling extensibility to multi-script support (Hindi, Marathi, Sanskrit).

## Architecture & Data Flow

### Core Pipeline
1. **Raw Data** (`data/devnagri-gold-dataset.jsonl`) → 2. **Format Conversion** (`DataConverter.jsonl_to_crf()`) → 3. **CRF Format** (`data/crf_train_data_full.txt`) → 4. **Training** (`DataLoader.load_crf_format()` + `CRFModel.train()`) → 5. **Feature Extraction** (`FeatureExtractor.extract_syllable_features()`) → 6. **Inference** (`SyllableSegmenter.segment_word()`)

### Critical Module Dependencies

- **`src/config.py`**: Single source of truth—never hardcode paths or constants
  - Paths: RAW_DATASET, CRF_TRAIN_DATA_FULL, MODEL_PATH, BILSTM_CRF_MODEL_PATH
  - Training params: MAX_ITER=200, TEST_SPLIT_RATIO=0.2, RANDOM_STATE=42
  - Devanagari constants: DEVANAGARI_CONSONANTS, VIRAMA, VOWEL_SIGNS
  
- **`src/crf_model.py`** (Phase 1): sklearn LogisticRegression + DictVectorizer
  - API: `train(sentences)` → dict metrics, `predict(features_list)`, `save(path)`, `load(path)`
  - Returns JSON-serializable metrics: precision, recall, f1, support
  
- **`src/bilstm_crf_model.py`** (Phase 2): PyTorch BiLSTM encoder + CRF layer
  - BiLSTMCRFTrainer class: `train(sentences, epochs=50)` with batch processing
  - Key components: embedding layer, BiLSTM, CRF transition scores
  - Saves as pkl; `SyllableSegmenter` auto-detects and loads both model types
  
- **`src/features.py`**: Linguistic feature extraction (shared across both models)
  - `FeatureExtractor.extract_syllable_features(syllable, sentence, index)` → dict
  - Features: syllable, length, prev/next_syllable, has_virama, has_vowel_sign, starts_with_consonant, is_short, is_long
  - All features JSON-serializable (bool/str/int only—critical for DictVectorizer)
  
- **`src/inference.py`**: Unified SyllableSegmenter (model-agnostic)
  - `segment_word(word: str) → List[str]`: Detects model type (CRF or BiLSTM+CRF) at runtime
  - Uses SyllableSplitter for initial chunking, then applies model refinement
  
- **`src/syllable_splitter.py`**: Hybrid chunking (lookup + rule-based fallback)
  - Caches training data in-memory on first call
  - Fallback: consonant + [virama] + [vowel mark] = syllable

### Data Format Details

**JSONL Input** (`devnagri-gold-dataset.jsonl`):
```json
{"word": "कर्म", "split": ["कर्", "म"]}
{"word": "नमस्ते", "split": ["न", "मस्", "ते"]}
```

**CRF Training Format** (`crf_train_data_full.txt`):
```
कर् B
म B
(blank line)
न B
मस् B
ते B
(blank line)
```
- Each syllable on a line with tag: `syllable TAG` (space-separated)
- TAG = `B` (boundary/start of word) or `O` (inside, not used currently)
- Blank line separates words
- DataLoader validates: must have 2 parts per line, no empty syllables/tags

## Coding Patterns & Conventions

### 1. Feature Engineering
- All features extracted via `FeatureExtractor.extract_syllable_features()` in `src/features.py`
- Features must be JSON-serializable (dict with bool/str/int values only—no custom objects)
- Add new features as new keys to the returned dict in `extract_syllable_features()`
- Context features use `<START>` and `<END>` tokens for word boundaries
- Test new features in `tests/test_features.py` with specific Devanagari examples before training

### 2. Configuration Management
- **Golden Rule**: Never hardcode paths/constants in model/inference code
- Always import from `src/config.py`: `from src.config import CRF_TRAIN_DATA_FULL, DEVANAGARI_VIRAMA, etc.`
- For multi-script (Phase 2), create language-specific configs (e.g., `config_hindi.py`) with same interface

### 3. Data Flow & Validation
- **Preprocessing**: `DataConverter.jsonl_to_crf()` converts JSONL → CRF format; tracks word count
- **Loading**: `DataLoader.load_crf_format()` validates structure (rsplit by last space, checks for 2 parts)
- **Augmentation**: `DataLoader.add_synthetic_negatives()` prevents class imbalance (if imbalance detected)
- **Never skip validation**—format errors cause silent failures in DictVectorizer

### 4. Model Interface Stability
- `CRFModel` has stable interface: `train(sentences)`, `predict(features_list)`, `predict_proba()`, `save(path)`, `load(path)`
- Methods return JSON-serializable dicts (metrics with precision/recall/f1/support keys)
- This design allows swapping LogisticRegression to pycrfsuite or BiLSTM without breaking `SyllableSegmenter`

### 5. Logging
- Initialize in each module: `logger = logging.getLogger(__name__)`
- Scripts configure INFO level; model training uses DEBUG for detailed steps
- Log major milestones: "Starting training", "Loaded N sentences", "✓ Model saved"
- Use `logger.warning()` for validation issues (malformed lines), `logger.error()` for exceptions

## Development Workflows

### Training CRF Model (Phase 1)
```bash
# 1. Preprocess (if raw JSONL changed)
python scripts/preprocess.py  # Converts JSONL → crf_train_data_full.txt

# 2. Train CRF baseline
python scripts/train.py  # Outputs: models/crf_model.pkl + models/metrics.json
```

### Training BiLSTM+CRF Model (Phase 2)
```bash
# Train neural model (requires PyTorch)
python scripts/train_bilstm_crf.py  # Outputs: models/bilstm_crf_model.pkl

# Inference auto-detects model type—no code changes needed
```

### Inference (Works with Either Model)
```bash
# Auto-detects CRF or BiLSTM+CRF based on model file
python scripts/infer.py "कर्म"

# With confidence scores (CRF model)
python scripts/infer.py "कर्म" --confidence

# Custom model path (any trained model)
python scripts/infer.py "कर्म" --model /path/to/model.pkl
```

### Testing
```bash
pytest tests/ -v --cov=src           # All tests with coverage
pytest tests/test_features.py -v     # Feature extraction
pytest tests/test_data.py -v         # Data loading & conversion
```

## Common Tasks & Decision Points

### Adding a New Feature
1. Define in `FeatureExtractor.extract_syllable_features()` (src/features.py)
   - Keep it JSON-serializable (bool/str/int only)
   - Document the linguistic rationale
2. Add unit test in `tests/test_features.py` with Devanagari examples
3. Retrain model: `python scripts/train.py`
4. Validate F1 impact in `models/metrics.json`
   - If F1 drops, the feature may be redundant or harmful

### Model Selection for Phase 2
- **CRF Model** (`src/crf_model.py`): Lightweight, interpretable, fast training
- **BiLSTM+CRF** (`src/bilstm_crf_model.py`): Better context modeling, handles longer dependencies
- Both models use identical `FeatureExtractor`; `SyllableSegmenter` auto-detects at runtime (check model file structure)
- Train both and compare `models/metrics.json` to decide which to use for inference

### Improving Model Performance
- **Low F1 scores**: Check `models/metrics.json` for precision/recall imbalance
  - If many false negatives: increase training data or add synthetic negatives (see `DataLoader.add_synthetic_negatives()`)
  - If many false positives: features may be overlapping (check vectorizer dims in metrics)
- **Model not converging**: Increase `MAX_ITER` in `src/config.py` if logs show "not converged"
- **Slow inference**: Profile SyllableSplitter—it caches training data on first call; could be optimized for Phase 3

### Multi-Script Extension (Phase 3 Design)
- Create `src/config_hindi.py`, `src/config_marathi.py` with script-specific constants (characters, rules)
- Modify `FeatureExtractor` to accept optional `language` parameter
- `DataConverter` and `DataLoader` are already language-agnostic
- Single model can be trained per-language or shared with language embeddings

## Key Files Reference

| File | Purpose | Modify When |
|------|---------|------------|
| `src/config.py` | Project constants & paths | Adding language support, changing thresholds |
| `src/features.py` | Feature engineering | New linguistic features needed |
| `src/crf_model.py` | Phase 1 model (LogReg+DictVec) | Implementing pycrfsuite instead |
| `src/bilstm_crf_model.py` | Phase 2 model (PyTorch) | Tuning architecture (LSTM size, CRF transitions) |
| `src/inference.py` | Model-agnostic segmentation API | Changing output format or adding batch processing |
| `scripts/train.py` | CRF training entry point | Changing training procedure (e.g., cross-validation) |
| `scripts/train_bilstm_crf.py` | BiLSTM+CRF training entry point | Adjusting epochs, learning rate, batch size |
| `src/syllable_splitter.py` | Initial chunking (lookup+rules) | Optimizing performance or supporting new scripts |

## Debugging Tips

**Model not converging**: Check MAX_ITER in config; increase if warning appears in logs
**Low F1 scores**: Validate data format (blank lines between words); run `pytest tests/test_data.py::TestDataLoader -v`
**Inference crashes**: Ensure model exists at path; verify word is valid Devanagari (not ASCII)
**BiLSTM model loads as CRF**: Check model file—BiLSTMCRFTrainer must have `.model` and `.model.forward` attributes
**Features seem redundant**: Check vectorizer output dimensions in metrics.json
**DictVectorizer errors**: All feature values must be bool/str/int/float—custom objects fail silently

## Critical Implementation Details

### Model Detection in SyllableSegmenter
```python
# Automatically detects model type at runtime
if hasattr(model, 'model') and hasattr(model.model, 'forward'):
    self.model_type = 'bilstm_crf'  # PyTorch model
else:
    self.model_type = 'crf'  # CRFModel instance
```
This allows swapping models without code changes to inference.

### Syllable Splitting Strategy
`SyllableSplitter.split()` uses hybrid two-stage approach:
1. **Lookup**: Check if word exists in training dataset (cached in memory)
2. **Fallback**: Linguistic rules—consonant + optional(virama) + optional(vowel_mark) = syllable
Ensures accuracy on seen words and robustness for new words.

### Tag System in CRF Format
Currently uses only `B` (boundary) tags. Each syllable tagged as `B` means "start of word unit". BiLSTM variant may upgrade to proper IOB tagging for richer sequence information.
