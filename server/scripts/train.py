#!/usr/bin/env python3
"""
Training script for CRF model.
Trains syllable segmentation model on preprocessed CRF data.

Usage:
    python scripts/train.py
"""

import json
import logging
from pathlib import Path

from src.config import CRF_TRAIN_DATA_FULL, MODEL_PATH, MODEL_METRICS_PATH
from src.data_loader import DataLoader
from src.crf_model import CRFModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train CRF model on syllable data."""
    logger.info("Starting CRF model training...")

    # Load data
    logger.info(f"Loading training data from {CRF_TRAIN_DATA_FULL}...")
    sentences = DataLoader.load_crf_format(CRF_TRAIN_DATA_FULL)
    logger.info(f"Loaded {len(sentences)} sentences")

    # Augment with synthetic negatives
    sentences = DataLoader.add_synthetic_negatives(sentences)

    # Train model
    model = CRFModel()
    train_results = model.train(sentences)

    # Save model
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    # Save metrics
    Path(MODEL_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(train_results, f, indent=2)

    logger.info(f"\n✓ Model saved to {MODEL_PATH}")
    logger.info(f"✓ Metrics saved to {MODEL_METRICS_PATH}")
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
