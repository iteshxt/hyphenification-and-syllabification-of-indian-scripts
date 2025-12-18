"""
Inference pipeline for Devanagari syllable segmentation.
Provides end-to-end syllable prediction for new words.
"""

import logging
from typing import List, Dict, Tuple, Optional

from src.crf_model import CRFModel
from src.features import FeatureExtractor
from src.syllable_splitter import SyllableSplitter

logger = logging.getLogger(__name__)


class SyllableSegmenter:
    """Predict syllable boundaries for Devanagari words."""

    def __init__(self, model_path: str):
        """
        Initialize segmenter with trained model.

        Args:
            model_path: Path to saved CRF model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.model = CRFModel.load(model_path)
        logger.info(f"Segmenter initialized with model from {model_path}")

    def _split_into_chunks(self, word: str) -> List[str]:
        """
        Split word into syllable chunks using SyllableSplitter.
        
        Args:
            word: Devanagari word
            
        Returns:
            List of syllable chunks
        """
        return SyllableSplitter.split(word)

    def segment_word(self, word: str) -> List[str]:
        """
        Segment a Devanagari word into syllables.

        Args:
            word: Devanagari word to segment

        Returns:
            List of syllables

        Raises:
            ValueError: If word is empty
        """
        if not word:
            raise ValueError("Word cannot be empty")

        # Split word into chunks
        chunks = self._split_into_chunks(word)
        
        # Extract features for each chunk
        features_list = []
        for i, chunk in enumerate(chunks):
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

            features = FeatureExtractor.extract_syllable_features(
                chunk, prev_chunk, next_chunk
            )
            features_list.append(features)

        # Get predictions
        predictions = self.model.predict(features_list)

        # Group chunks into syllables based on predictions
        syllables = []
        current_syllable = ""

        for chunk, pred in zip(chunks, predictions):
            if pred == "B" and current_syllable:
                # Start of new syllable
                syllables.append(current_syllable)
                current_syllable = chunk
            else:
                # Continue current syllable
                current_syllable += chunk

        # Add final syllable
        if current_syllable:
            syllables.append(current_syllable)

        return syllables

    def segment_word_with_confidence(
        self, word: str
    ) -> List[Dict[str, str | float]]:
        """
        Segment word and return confidence scores.

        Args:
            word: Devanagari word to segment

        Returns:
            List of syllables with confidence scores

        Raises:
            ValueError: If word is empty
        """
        if not word:
            raise ValueError("Word cannot be empty")

        # Split word into chunks
        chunks = self._split_into_chunks(word)
        
        # Extract features
        features_list = []
        for i, chunk in enumerate(chunks):
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

            features = FeatureExtractor.extract_syllable_features(
                chunk, prev_chunk, next_chunk
            )
            features_list.append(features)

        # Get predictions and probabilities
        predictions = self.model.predict(features_list)
        probabilities = self.model.predict_proba(features_list)

        # Get class labels
        classes = self.model.model.classes_
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Reconstruct syllables with confidence
        syllables_with_conf = []
        current_syllable = ""
        max_confidence = 0.0

        for i, (chunk, pred) in enumerate(zip(chunks, predictions)):
            # Get confidence for this prediction
            pred_idx = class_to_idx[pred]
            confidence = probabilities[i][pred_idx]

            if pred == "B" and current_syllable:
                # Start of new syllable
                syllables_with_conf.append(
                    {"syllable": current_syllable, "confidence": max_confidence}
                )
                current_syllable = chunk
                max_confidence = confidence
            else:
                # Continue current syllable
                current_syllable += chunk
                max_confidence = max(max_confidence, confidence)

        # Add final syllable
        if current_syllable:
            syllables_with_conf.append(
                {"syllable": current_syllable, "confidence": max_confidence}
            )

        return syllables_with_conf

    def batch_segment(self, words: List[str]) -> Dict[str, List[str]]:
        """
        Segment multiple words.

        Args:
            words: List of Devanagari words

        Returns:
            Dictionary mapping words to their syllable segments
        """
        results = {}
        for word in words:
            try:
                results[word] = self.segment_word(word)
            except Exception as e:
                logger.error(f"Error segmenting word '{word}': {e}")
                results[word] = None

        return results
