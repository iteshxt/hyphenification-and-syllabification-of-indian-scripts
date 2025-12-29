"""
Data loading and preprocessing utilities for CRF training.
Handles JSONL and CRF format data with validation and error handling.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess syllabification training data."""

    @staticmethod
    def load_crf_format(filepath: str) -> List[List[Tuple[str, str]]]:
        """
        Load CRF format data (syllable TAG pairs separated by blank lines).

        Args:
            filepath: Path to CRF format file

        Returns:
            List of sentences, where each sentence is a list of (syllable, tag) tuples

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or malformed
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        sentences = []
        current_sentence = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    if not line:
                        # Blank line = sentence boundary
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                    else:
                        # Parse "syllable TAG" format
                        parts = line.rsplit(" ", 1)
                        if len(parts) != 2:
                            logger.warning(
                                f"Line {line_num}: Invalid format (expected 'syllable TAG'), "
                                f"got '{line}'"
                            )
                            continue

                        syllable, tag = parts
                        if not syllable or not tag:
                            logger.warning(
                                f"Line {line_num}: Empty syllable or tag, skipping"
                            )
                            continue

                        current_sentence.append((syllable, tag))

            # Add final sentence if exists
            if current_sentence:
                sentences.append(current_sentence)

            if not sentences:
                raise ValueError("No valid sentences found in CRF data file")

            logger.info(f"Loaded {len(sentences)} sentences from {filepath}")
            return sentences

        except Exception as e:
            logger.error(f"Error loading CRF data: {e}")
            raise

    @staticmethod
    def load_jsonl_dataset(filepath: str) -> List[Dict[str, any]]:
        """
        Load JSONL format dataset (one JSON object per line).

        Expected format: {"word": "कर्म", "split": ["क", "र्म"], "lang": "deva"}

        Args:
            filepath: Path to JSONL file

        Returns:
            List of dataset entries (dictionaries)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        entries = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)

                        # Validate required fields
                        if "word" not in entry or "split" not in entry:
                            logger.warning(
                                f"Line {line_num}: Missing 'word' or 'split' field"
                            )
                            continue

                        entries.append(entry)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON: {e}")
                        continue

            if not entries:
                raise ValueError("No valid entries found in JSONL dataset")

            logger.info(f"Loaded {len(entries)} entries from {filepath}")
            return entries

        except Exception as e:
            logger.error(f"Error loading JSONL dataset: {e}")
            raise

    @staticmethod
    def add_synthetic_negatives(
        sentences: List[List[Tuple[str, str]]]
    ) -> List[List[Tuple[str, str]]]:
        """
        Augment training data by adding synthetic negative examples.

        Concatenates adjacent syllables to create non-boundary examples (I tags).

        Args:
            sentences: Original CRF sentences

        Returns:
            Expanded sentences with synthetic negatives
        """
        expanded_sentences = []

        for sent in sentences:
            expanded = []

            for i, (syll, tag) in enumerate(sent):
                expanded.append((syll, tag))  # Original positive

                # Create synthetic negative: concatenate with next syllable
                if i < len(sent) - 1:
                    next_syll = sent[i + 1][0]
                    combined = syll + next_syll
                    expanded.append((combined, "I"))  # Synthetic non-boundary

            expanded_sentences.append(expanded)

        logger.info(
            f"Augmented {len(sentences)} sentences with synthetic negatives "
            f"(from {sum(len(s) for s in sentences)} to "
            f"{sum(len(s) for s in expanded_sentences)} syllables)"
        )

        return expanded_sentences
