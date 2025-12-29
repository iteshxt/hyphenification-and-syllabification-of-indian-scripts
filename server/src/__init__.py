"""
Devanagari Syllabification - Core package.

Modules:
    - config: Configuration management
    - data_loader: Data loading utilities
    - data_converter: Format conversion utilities
    - features: Feature extraction
    - crf_model: CRF model implementation
    - inference: Inference pipeline
"""

from src.config import *
from src.crf_model import CRFModel
from src.inference import SyllableSegmenter

__version__ = "0.1.0"
__author__ = "Aditya"

__all__ = ["CRFModel", "SyllableSegmenter"]
