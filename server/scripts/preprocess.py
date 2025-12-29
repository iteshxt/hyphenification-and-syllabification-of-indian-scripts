#!/usr/bin/env python3
"""
Data preprocessing script.
Convert raw JSONL dataset to CRF training format.

Usage:
    python scripts/preprocess.py
"""

import logging

from src.config import RAW_DATASET, CRF_TRAIN_DATA_FULL
from src.data_converter import DataConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Convert JSONL to CRF format."""
    logger.info("Starting data preprocessing...")
    logger.info(f"Input: {RAW_DATASET}")
    logger.info(f"Output: {CRF_TRAIN_DATA_FULL}\n")

    try:
        word_count = DataConverter.jsonl_to_crf(RAW_DATASET, CRF_TRAIN_DATA_FULL)
        logger.info(f"\n✓ Preprocessing complete: {word_count} words converted")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")


if __name__ == "__main__":
    main()
