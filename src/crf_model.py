"""
CRF-based model for Devanagari syllable segmentation.
Uses sklearn's LogisticRegression as a CRF proxy with feature vectorization.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.config import RANDOM_STATE, TEST_SPLIT_RATIO, MAX_ITER
from src.features import extract_sentence_features

logger = logging.getLogger(__name__)


class CRFModel:
    """Conditional Random Field model for syllable boundary detection."""

    def __init__(self, max_iter: int = MAX_ITER, random_state: int = RANDOM_STATE):
        """
        Initialize CRF model.

        Args:
            max_iter: Maximum iterations for LogisticRegression training
            random_state: Random seed for reproducibility
        """
        self.model = LogisticRegression(
            max_iter=max_iter, random_state=random_state, n_jobs=-1
        )
        self.vectorizer = DictVectorizer()
        self.is_trained = False

    def train(
        self,
        sentences: List[List[Tuple[str, str]]],
        test_size: float = TEST_SPLIT_RATIO,
    ) -> Dict[str, Any]:
        """
        Train CRF model on syllable data.

        Args:
            sentences: List of (syllable, tag) pairs
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary containing model metrics and train/test info
        """
        logger.info(f"Training CRF model on {len(sentences)} sentences...")

        # Extract features and labels
        X_all = [extract_sentence_features(sent) for sent in sentences]
        y_all = [[tag for (_, tag) in sent] for sent in sentences]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size, random_state=RANDOM_STATE
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Flatten for sklearn
        X_train_flat = [f for sent in X_train for f in sent]
        y_train_flat = [label for sent in y_train for label in sent]
        X_test_flat = [f for sent in X_test for f in sent]
        y_test_flat = [label for sent in y_test for label in sent]

        # Fit vectorizer and vectorize
        logger.info("Vectorizing features...")
        X_train_vec = self.vectorizer.fit_transform(X_train_flat)
        X_test_vec = self.vectorizer.transform(X_test_flat)

        logger.info(f"Feature dimension: {X_train_vec.shape[1]}")

        # Train model
        logger.info("Training LogisticRegression model...")
        self.model.fit(X_train_vec, y_train_flat)

        # Evaluate
        logger.info("Evaluating model...")
        y_pred_flat = self.model.predict(X_test_vec)

        # Reconstruct sentence-level predictions
        y_pred = []
        idx = 0
        for sent in y_test:
            sent_len = len(sent)
            y_pred.append(y_pred_flat[idx : idx + sent_len])
            idx += sent_len

        metrics = self._evaluate(y_test_flat, y_pred_flat, y_test, y_pred)

        self.is_trained = True
        logger.info("Model training complete!")

        return {
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": X_train_vec.shape[1],
        }

    def predict(self, features_list: List[Dict]) -> np.ndarray:
        """
        Predict labels for a list of features.

        Args:
            features_list: List of feature dictionaries

        Returns:
            Array of predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X_vec = self.vectorizer.transform(features_list)
        return self.model.predict(X_vec)

    def predict_proba(self, features_list: List[Dict]) -> np.ndarray:
        """
        Get probability predictions for a list of features.

        Args:
            features_list: List of feature dictionaries

        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X_vec = self.vectorizer.transform(features_list)
        return self.model.predict_proba(X_vec)

    def save(self, filepath: str) -> None:
        """
        Save trained model and vectorizer to file.

        Args:
            filepath: Path to save model to
        """
        if not self.is_trained:
            logger.warning("Model not trained. Saving untrained model.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CRFModel":
        """
        Load trained model from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded CRFModel instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        crf = cls()
        crf.model = model_data["model"]
        crf.vectorizer = model_data["vectorizer"]
        crf.is_trained = model_data.get("is_trained", True)

        logger.info(f"Model loaded from {filepath}")
        return crf

    @staticmethod
    def _evaluate(
        y_test_flat: List[str],
        y_pred_flat: np.ndarray,
        y_test: List[List[str]],
        y_pred: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            y_test_flat: Flattened true labels
            y_pred_flat: Flattened predictions
            y_test: Sentence-level true labels
            y_pred: Sentence-level predictions

        Returns:
            Dictionary of metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_flat, y_pred_flat, average="weighted"
        )

        correct = sum(1 for i, j in zip(y_test_flat, y_pred_flat) if i == j)
        accuracy = correct / len(y_test_flat)

        logger.info(f"\n{'='*60}")
        logger.info("CRF MODEL EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"\nWeighted Metrics:")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  Accuracy:  {accuracy:.4f} ({correct}/{len(y_test_flat)})")
        logger.info(f"\nPer-Class Report:")
        logger.info(f"\n{classification_report(y_test_flat, y_pred_flat)}")

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }
