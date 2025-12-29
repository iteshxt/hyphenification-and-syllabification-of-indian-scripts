"""
Tests for data loading and conversion utilities.
Run with: pytest tests/test_data.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data_loader import DataLoader
from src.data_converter import DataConverter


class TestDataConverter:
    """Test JSONL to CRF conversion."""

    def test_jsonl_to_crf_conversion(self):
        """Test basic JSONL to CRF conversion."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write test data
            json.dump({"word": "कर्म", "split": ["कर्", "म"]}, f)
            f.write("\n")
            json.dump({"word": "नमस्ते", "split": ["न", "मस्", "ते"]}, f)
            input_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_file = f.name

        try:
            # Convert
            word_count = DataConverter.jsonl_to_crf(input_file, output_file)

            # Verify
            assert word_count == 2

            # Check output format
            with open(output_file, "r") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert "कर् B" in lines
            assert "म B" in lines
            assert "" in lines  # Blank line separators

        finally:
            Path(input_file).unlink()
            Path(output_file).unlink()

    def test_jsonl_to_crf_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            DataConverter.jsonl_to_crf(
                "/nonexistent/path.jsonl", "/tmp/output.txt"
            )


class TestDataLoader:
    """Test CRF data loading."""

    def test_load_crf_format(self):
        """Test loading CRF format data."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            # Write test data
            f.write("कर् B\n")
            f.write("म B\n")
            f.write("\n")
            f.write("नमस्ते B\n")
            filepath = f.name

        try:
            sentences = DataLoader.load_crf_format(filepath)

            # Verify structure
            assert len(sentences) == 2
            assert len(sentences[0]) == 2  # First word has 2 syllables
            assert len(sentences[1]) == 1  # Second word has 1 syllable

            # Verify content
            assert sentences[0][0] == ("कर्", "B")
            assert sentences[0][1] == ("म", "B")
            assert sentences[1][0] == ("नमस्ते", "B")

        finally:
            Path(filepath).unlink()

    def test_load_crf_format_file_not_found(self):
        """Test error for missing file."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_crf_format("/nonexistent/data.txt")

    def test_add_synthetic_negatives(self):
        """Test synthetic negative example generation."""
        sentences = [
            [("क", "B"), ("र्म", "B")],
        ]

        augmented = DataLoader.add_synthetic_negatives(sentences)

        # Should have original + synthetic examples
        assert len(augmented[0]) > len(sentences[0])

        # Check for synthetic negative (concatenated with I tag)
        tags = [tag for _, tag in augmented[0]]
        assert "I" in tags


class TestDataLoaderJSONL:
    """Test JSONL data loading."""

    def test_load_jsonl_dataset(self):
        """Test loading JSONL dataset."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"word": "कर्म", "split": ["कर्", "म"]}, f)
            f.write("\n")
            json.dump({"word": "नमस्ते", "split": ["न", "मस्", "ते"]}, f)
            filepath = f.name

        try:
            entries = DataLoader.load_jsonl_dataset(filepath)

            assert len(entries) == 2
            assert entries[0]["word"] == "कर्म"
            assert entries[1]["word"] == "नमस्ते"

        finally:
            Path(filepath).unlink()

    def test_load_jsonl_invalid_json(self):
        """Test handling of malformed JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            # Valid entry
            json.dump({"word": "कर्म", "split": ["कर्", "म"]}, f)
            f.write("\n")
            # Invalid JSON
            f.write("{ this is not json }\n")
            # Valid entry again
            json.dump({"word": "नमस्ते", "split": ["न", "मस्", "ते"]}, f)
            filepath = f.name

        try:
            entries = DataLoader.load_jsonl_dataset(filepath)

            # Should load only valid entries (skipping malformed one)
            assert len(entries) == 2

        finally:
            Path(filepath).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
