"""
Feature extraction for Devanagari syllable segmentation.
Provides utilities for extracting linguistic and contextual features from syllables.
"""

from typing import Dict, List, Tuple, Optional
from src.config import (
    DEVANAGARI_CONSONANTS,
    DEVANAGARI_VOWEL_SIGNS,
    DEVANAGARI_VIRAMA,
    SHORT_SYLLABLE_THRESHOLD,
    LONG_SYLLABLE_THRESHOLD,
)


class FeatureExtractor:
    """Extract features for syllable boundary detection."""

    @staticmethod
    def extract_syllable_features(
        syllable: str,
        prev_syllable: Optional[str] = None,
        next_syllable: Optional[str] = None,
    ) -> Dict[str, bool | str | int]:
        """
        Extract linguistic features from a syllable.

        Args:
            syllable: The syllable to extract features from
            prev_syllable: Previous syllable for context
            next_syllable: Next syllable for context

        Returns:
            Dictionary of feature names and values
        """
        prev_label = prev_syllable if prev_syllable else "<START>"
        next_label = next_syllable if next_syllable else "<END>"

        features: Dict[str, bool | str | int] = {
            # Lexical features
            "syllable": syllable,
            "syllable.length": len(syllable),
            # Context features
            "prev_syllable": prev_label,
            "next_syllable": next_label,
            # Morphological features
            "has_virama": DEVANAGARI_VIRAMA in syllable,
            "has_vowel_sign": any(c in syllable for c in DEVANAGARI_VOWEL_SIGNS),
            "starts_with_consonant": (
                syllable[0] in DEVANAGARI_CONSONANTS if syllable else False
            ),
            # Structural features
            "is_short": len(syllable) <= SHORT_SYLLABLE_THRESHOLD,
            "is_long": len(syllable) > LONG_SYLLABLE_THRESHOLD,
        }

        return features

    @staticmethod
    def extract_sequence_features(
        sequence: List[Tuple[str, Optional[str], Optional[str]]]
    ) -> List[Dict[str, bool | str | int]]:
        """
        Extract features for a sequence of syllables.

        Args:
            sequence: List of (syllable, prev_syllable, next_syllable) tuples

        Returns:
            List of feature dictionaries
        """
        features_list = []
        for syllable, prev_syll, next_syll in sequence:
            features = FeatureExtractor.extract_syllable_features(
                syllable, prev_syll, next_syll
            )
            features_list.append(features)

        return features_list


def extract_sentence_features(sent: List[Tuple[str, str]]) -> List[Dict]:
    """
    Extract features for a sentence (list of syllable-label pairs).

    Args:
        sent: List of (syllable, label) tuples

    Returns:
        List of feature dictionaries
    """
    features_list = []

    for i, (syllable, label) in enumerate(sent):
        prev_syll = sent[i - 1][0] if i > 0 else None
        next_syll = sent[i + 1][0] if i < len(sent) - 1 else None

        features = FeatureExtractor.extract_syllable_features(
            syllable, prev_syll, next_syll
        )
        features_list.append(features)

    return features_list
