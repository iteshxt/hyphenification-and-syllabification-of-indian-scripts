"""
Unit tests for syllabification package.
Run with: pytest tests/ -v
"""

import pytest
from src.features import FeatureExtractor, extract_sentence_features


class TestFeatureExtractor:
    """Test feature extraction functionality."""

    def test_extract_syllable_features_basic(self):
        """Test basic feature extraction."""
        features = FeatureExtractor.extract_syllable_features(
            "कर्", prev_syllable="क", next_syllable="म"
        )

        assert features["syllable"] == "कर्"
        assert features["syllable.length"] == 2
        assert features["prev_syllable"] == "क"
        assert features["next_syllable"] == "म"
        assert features["has_virama"] is True

    def test_extract_syllable_features_with_vowel_sign(self):
        """Test feature extraction with vowel sign."""
        features = FeatureExtractor.extract_syllable_features("वि")

        assert features["has_vowel_sign"] is True

    def test_extract_syllable_features_start_end_tokens(self):
        """Test feature extraction with start/end tokens."""
        features = FeatureExtractor.extract_syllable_features("कर्", None, "म")

        assert features["prev_syllable"] == "<START>"
        assert features["next_syllable"] == "म"

    def test_extract_syllable_features_consonant_detection(self):
        """Test consonant detection."""
        features = FeatureExtractor.extract_syllable_features("क")
        assert features["starts_with_consonant"] is True

        features = FeatureExtractor.extract_syllable_features("ा")
        assert features["starts_with_consonant"] is False

    def test_extract_sequence_features(self):
        """Test sequence feature extraction."""
        sequence = [
            ("कर्", None, "म"),
            ("म", "कर्", None),
        ]

        features_list = FeatureExtractor.extract_sequence_features(sequence)

        assert len(features_list) == 2
        assert all(isinstance(f, dict) for f in features_list)
        assert features_list[0]["syllable"] == "कर्"
        assert features_list[1]["syllable"] == "म"


class TestSentenceFeatures:
    """Test sentence-level feature extraction."""

    def test_extract_sentence_features(self):
        """Test extracting features from sentence."""
        sent = [("कर्", "B"), ("म", "B")]
        features = extract_sentence_features(sent)

        assert len(features) == 2
        assert features[0]["prev_syllable"] == "<START>"
        assert features[0]["next_syllable"] == "म"
        assert features[1]["prev_syllable"] == "कर्"
        assert features[1]["next_syllable"] == "<END>"

    def test_extract_sentence_features_empty(self):
        """Test with empty sentence."""
        features = extract_sentence_features([])
        assert features == []

    def test_extract_sentence_features_single_syllable(self):
        """Test with single syllable."""
        sent = [("कर्म", "B")]
        features = extract_sentence_features(sent)

        assert len(features) == 1
        assert features[0]["syllable"] == "कर्म"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
