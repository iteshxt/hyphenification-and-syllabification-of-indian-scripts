#!/usr/bin/env python3
"""
Train BiLSTM+CRF model for Devanagari syllable segmentation.
Saves the trained model as pkl file.
"""

import sys
import logging
from pathlib import Path
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CRF_TRAIN_DATA_FULL, BILSTM_CRF_MODEL_PATH
from src.data_loader import DataLoader
from src.features import FeatureExtractor
from src.bilstm_crf_model import BiLSTMCRFTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Training BiLSTM+CRF Model")
    logger.info("=" * 60)
    
    # Load training data
    logger.info(f"Loading data from {CRF_TRAIN_DATA_FULL}")
    sentences = DataLoader.load_crf_format(CRF_TRAIN_DATA_FULL)
    logger.info(f"✓ Loaded {len(sentences)} sentences")
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = FeatureExtractor()
    sentences_with_features = []
    
    for sentence in sentences:
        features = []
        for i, syllable in enumerate(sentence):
            feat = feature_extractor.extract_syllable_features(
                syllable, sentence, i
            )
            features.append(feat)
        sentences_with_features.append(features)
    
    logger.info(f"✓ Features extracted for {len(sentences_with_features)} sentences")
    
    # Build character vocabulary
    char_vocab = set()
    for sentence in sentences:
        for syllable in sentence:
            for char in syllable:
                char_vocab.add(char)
    vocab_size = len(char_vocab) + 2  # +2 for padding and unknown
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Train model
    logger.info("Training BiLSTM+CRF model...")
    trainer = BiLSTMCRFTrainer(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        learning_rate=0.001,
        batch_size=32
    )
    
    try:
        metrics = trainer.train(sentences, epochs=50)
        
        logger.info("✓ Training completed!")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key.capitalize()}: {value:.4f}")
        
        # Save model
        model_path = Path(BILSTM_CRF_MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {BILSTM_CRF_MODEL_PATH}")
        with open(BILSTM_CRF_MODEL_PATH, 'wb') as f:
            pickle.dump(trainer, f)
        
        logger.info(f"✓ Model saved successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
