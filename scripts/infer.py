#!/usr/bin/env python3
"""
Inference script for syllable segmentation.
Segment Devanagari words using trained model.

Usage:
    python scripts/infer.py "कर्म"
    python scripts/infer.py "कर्म" "विद्यालय" --confidence
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import MODEL_PATH
from src.inference import SyllableSegmenter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run inference on provided words."""
    parser = argparse.ArgumentParser(
        description="Segment Devanagari words into syllables"
    )
    parser.add_argument(
        "words",
        nargs="+",
        help="Devanagari words to segment",
    )
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Show confidence scores for each syllable",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to trained model (default: {MODEL_PATH})",
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found at {args.model}")
        logger.error("Please run 'python scripts/train.py' first")
        sys.exit(1)

    # Load segmenter
    try:
        segmenter = SyllableSegmenter(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Process words
    print("\n" + "=" * 70)
    print("DEVANAGARI SYLLABLE SEGMENTATION")
    print("=" * 70)

    for word in args.words:
        try:
            if args.confidence:
                result = segmenter.segment_word_with_confidence(word)
                print(f"\nWord: {word}")
                print(f"Segments:")
                for item in result:
                    print(
                        f"  {item['syllable']:15} (confidence: {item['confidence']:.4f})"
                    )
                print(f"Result: {' + '.join([r['syllable'] for r in result])}")
            else:
                syllables = segmenter.segment_word(word)
                print(f"\nWord: {word}")
                print(f"Segments: {' + '.join(syllables)}")

        except Exception as e:
            logger.error(f"Error processing '{word}': {e}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
