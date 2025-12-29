"""
Configuration management for Devanagari Syllabification project.
Centralized constants and settings for reproducibility and maintainability.
"""

from pathlib import Path
from typing import Final

# ==================== PROJECT PATHS ====================
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
SCRIPTS_DIR: Final[Path] = PROJECT_ROOT / "scripts"

# ==================== DATA FILES ====================
RAW_DATASET: Final[str] = str(DATA_DIR / "devnagri-gold-dataset.jsonl")
CRF_TRAIN_DATA: Final[str] = str(DATA_DIR / "crf_train_data.txt")
CRF_TRAIN_DATA_FULL: Final[str] = str(DATA_DIR / "crf_train_data_full.txt")

# ==================== MODEL FILES ====================
MODEL_PATH: Final[str] = str(MODELS_DIR / "crf_model.pkl")
BILSTM_CRF_MODEL_PATH: Final[str] = str(MODELS_DIR / "bilstm_crf_model.pkl")
MODEL_METRICS_PATH: Final[str] = str(MODELS_DIR / "metrics.json")

# ==================== TRAINING PARAMETERS ====================
RANDOM_STATE: Final[int] = 42
TEST_SPLIT_RATIO: Final[float] = 0.2
MAX_ITER: Final[int] = 200

# ==================== DEVANAGARI SCRIPT CONSTANTS ====================
DEVANAGARI_CONSONANTS: Final[str] = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
DEVANAGARI_VOWEL_SIGNS: Final[str] = 'ािीुूेैोौ'
DEVANAGARI_VIRAMA: Final[str] = '्'

# ==================== FEATURE EXTRACTION ====================
SHORT_SYLLABLE_THRESHOLD: Final[int] = 2
LONG_SYLLABLE_THRESHOLD: Final[int] = 4

# ==================== LOGGING & OUTPUT ====================
VERBOSE: Final[bool] = True
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
