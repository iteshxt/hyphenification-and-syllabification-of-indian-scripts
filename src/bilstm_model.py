"""
BiLSTM model for Devanagari syllable segmentation.
Simple neural network baseline for comparison with CRF.

Architecture:
- Character embeddings (learned)
- Bidirectional LSTM
- Softmax output for syllable boundary tags
"""

import logging
import pickle
from typing import List, Tuple, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.config import RANDOM_STATE, TEST_SPLIT_RATIO, MAX_ITER

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class BiLSTMModel(nn.Module):
    """BiLSTM model for syllable boundary detection."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize BiLSTM model.

        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden dimension of LSTM
            output_dim: Output dimension (2 for binary tags: B/O)
            dropout: Dropout probability
        """
        super(BiLSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if embedding_dim > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len)
            lengths: List of sequence lengths for packing

        Returns:
            Output logits of shape (batch_size, seq_len, output_dim)
        """
        # Embed
        embedded = self.dropout(self.embedding(x))

        # Pack sequences
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        lstm_out, _ = self.lstm(packed)

        # Unpack
        output, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Linear layer
        logits = self.fc(output)

        return logits


class BiLSTMTrainer:
    """Trainer for BiLSTM model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        """Initialize trainer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = BiLSTMModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.tag_to_idx = {"O": 0, "B": 1}
        self.idx_to_tag = {0: "O", 1: "B"}

    def _syllables_to_sequences(self, sentences: List[List[Tuple[str, str]]]) -> Tuple[List, List]:
        """Convert syllables to character sequences."""
        char_vocab = set()
        for sent in sentences:
            for syllable, _ in sent:
                char_vocab.update(syllable)

        char_vocab = sorted(list(char_vocab))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(char_vocab)}  # 0 is padding
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        sequences = []
        labels = []

        for sent in sentences:
            seq = []
            tag_seq = []
            for syllable, tag in sent:
                char_seq = [self.char_to_idx.get(c, 0) for c in syllable]
                seq.extend(char_seq)
                # Each character in a syllable gets the same tag as the syllable
                tag_seq.extend([self.tag_to_idx[tag]] * len(char_seq))

            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(tag_seq, dtype=torch.long))

        return sequences, labels

    def train(
        self,
        sentences: List[List[Tuple[str, str]]],
        epochs: int = 50,
        test_size: float = TEST_SPLIT_RATIO,
    ) -> Dict[str, Any]:
        """
        Train BiLSTM model.

        Args:
            sentences: List of (syllable, tag) pairs
            epochs: Number of training epochs
            test_size: Fraction for test set

        Returns:
            Dictionary with metrics
        """
        logger.info(f"Preparing data for BiLSTM training...")

        # Convert to sequences
        sequences, labels = self._syllables_to_sequences(sentences)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=RANDOM_STATE
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Character vocab size: {len(self.char_to_idx)}")

        # Training loop
        logger.info(f"Training for {epochs} epochs...")
        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            batch_count = 0

            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                batch_seqs = X_train[i : i + self.batch_size]
                batch_labels = y_train[i : i + self.batch_size]

                # Pad sequences
                lengths = torch.tensor([len(seq) for seq in batch_seqs])
                padded_seqs = pad_sequence(batch_seqs, batch_first=True)
                padded_labels = pad_sequence(batch_labels, batch_first=True)

                padded_seqs = padded_seqs.to(self.device)
                padded_labels = padded_labels.to(self.device)

                # Forward
                logits = self.model(padded_seqs, lengths.to(self.device))
                loss = self.criterion(
                    logits.view(-1, 2), padded_labels.view(-1)
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluate
        logger.info("Evaluating BiLSTM model...")
        y_pred_flat = []
        y_test_flat = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test), self.batch_size):
                batch_seqs = X_test[i : i + self.batch_size]
                batch_labels = y_test[i : i + self.batch_size]

                lengths = torch.tensor([len(seq) for seq in batch_seqs])
                padded_seqs = pad_sequence(batch_seqs, batch_first=True)
                padded_labels = pad_sequence(batch_labels, batch_first=True)

                padded_seqs = padded_seqs.to(self.device)
                logits = self.model(padded_seqs, lengths.to(self.device))
                preds = torch.argmax(logits, dim=2)

                # Collect predictions (remove padding)
                for pred, label, length in zip(
                    preds.cpu(), padded_labels, lengths
                ):
                    y_pred_flat.extend(pred[:length].numpy())
                    y_test_flat.extend(label[:length].numpy())

        # Compute metrics
        y_pred_flat = np.array(y_pred_flat)
        y_test_flat = np.array(y_test_flat)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_flat, y_pred_flat, average="weighted"
        )

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(np.mean(y_pred_flat == y_test_flat)),
        }

        logger.info(f"BiLSTM Metrics: {metrics}")

        return {
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "char_vocab_size": len(self.char_to_idx),
            "epochs": epochs,
        }

    def predict(self, sequences: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Predict on sequences.

        Args:
            sequences: List of character sequences

        Returns:
            List of predicted tag sequences
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for seq in sequences:
                seq = seq.unsqueeze(0).to(self.device)
                lengths = torch.tensor([seq.shape[1]])
                logits = self.model(seq, lengths.to(self.device))
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                predictions.append(preds[0])

        return predictions

    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")
