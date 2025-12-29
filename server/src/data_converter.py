"""
Data conversion utilities for transforming between JSONL and CRF formats.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class DataConverter:
    """Convert between different data formats."""

    @staticmethod
    def jsonl_to_crf(input_file: str, output_file: str) -> int:
        """
        Convert JSONL dataset to CRF training format.

        Expected JSONL format: {"word": "कर्म", "split": ["क", "र्म"], "lang": "deva"}
        Output CRF format:
          - Each line: syllable TAG
          - TAG = "B" for syllable boundary
          - Blank line separates words

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output CRF file

        Returns:
            Number of words converted

        Raises:
            FileNotFoundError: If input file doesn't exist
        """
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        crf_lines = []
        word_count = 0

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        word = entry.get("word", "")
                        split = entry.get("split", [])

                        if not word or not split:
                            logger.warning(
                                f"Line {line_num}: Missing 'word' or 'split' field"
                            )
                            continue

                        # Convert split to CRF format (each syllable = B tag)
                        for syllable in split:
                            crf_lines.append(f"{syllable} B")

                        # Blank line separator
                        crf_lines.append("")
                        word_count += 1

                        if word_count % 100 == 0:
                            logger.info(f"Processed {word_count} words...")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: JSON error: {e}")
                        continue

            # Write output
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(crf_lines))

            logger.info(f"✓ Converted {word_count} words to CRF format")
            logger.info(f"  Output: {output_file} ({len(crf_lines)} lines)")

            return word_count

        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            raise
