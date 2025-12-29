"""
BiLSTM+CRF model for Devanagari syllable segmentation.
Combines BiLSTM encoder with CRF sequence tagging layer.

This is a more sophisticated model that considers tag transitions.
"""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from src.config import RANDOM_STATE, TEST_SPLIT_RATIO

logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class CRFLayer(nn.Module):
    """Conditional Random Field layer for sequence tagging."""

    def __init__(self, num_tags: int):
        """Initialize CRF layer."""
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags
        
        # Transition scores (from_tag, to_tag)
        self.transitions = nn.Parameter(
            torch.randn(num_tags, num_tags)
        )
        
        # Start and end tags
        self.start_tag = num_tags
        self.end_tag = num_tags + 1

    def forward(self, logits, tags, lengths):
        """
        Compute CRF loss.

        Args:
            logits: Tensor of shape (batch_size, seq_len, num_tags)
            tags: Tensor of shape (batch_size, seq_len)
            lengths: Tensor of sequence lengths

        Returns:
            CRF loss
        """
        return self._neg_log_likelihood(logits, tags, lengths)

    def _neg_log_likelihood(self, logits, tags, lengths):
        """Compute negative log likelihood for CRF."""
        batch_size, seq_len, num_tags = logits.shape
        
        # Forward path (alpha)
        forward_var = self._forward_alg(logits, lengths)
        
        # Gold path
        gold_score = self._score_sentence(logits, tags, lengths)
        
        # Loss = forward_var - gold_score
        return torch.mean(forward_var - gold_score)

    def _forward_alg(self, logits, lengths):
        """Compute forward algorithm (dynamic programming)."""
        batch_size, seq_len, num_tags = logits.shape
        
        # Initialize
        viterbi = logits[:, 0, :]  # (batch_size, num_tags)
        
        for t in range(1, seq_len):
            # Expand for broadcasting
            emit_scores = logits[:, t, :].unsqueeze(1)  # (batch_size, 1, num_tags)
            trans_scores = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            
            # Max path scores
            next_scores = viterbi.unsqueeze(2) + emit_scores + trans_scores
            viterbi = torch.logsumexp(next_scores, dim=1)
        
        return torch.logsumexp(viterbi, dim=1)

    def _score_sentence(self, logits, tags, lengths):
        """Score the gold sequence."""
        batch_size, seq_len, num_tags = logits.shape
        
        score = torch.zeros(batch_size, device=logits.device)
        
        for b in range(batch_size):
            length = lengths[b]
            for t in range(length):
                from_tag = int(tags[b, t].item())
                emit_score = logits[b, t, from_tag]
                
                if t > 0:
                    prev_tag = int(tags[b, t-1].item())
                    trans_score = self.transitions[prev_tag, from_tag]
                    score[b] += trans_score + emit_score
                else:
                    score[b] += emit_score
        
        return score

    def decode(self, logits, lengths):
        """Viterbi decoding."""
        batch_size, seq_len, num_tags = logits.shape
        
        # Viterbi
        viterbi = logits[:, 0, :]  # (batch_size, num_tags)
        backpointers = []
        
        for t in range(1, seq_len):
            next_scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            viterbi, bp = torch.max(next_scores, dim=1)
            viterbi = viterbi + logits[:, t, :]
            backpointers.append(bp)
        
        # Backtrack
        batch_paths = []
        for b in range(batch_size):
            path = [torch.argmax(viterbi[b]).item()]
            for bp_t in reversed(backpointers):
                path.append(bp_t[b, path[-1]].item())
            batch_paths.append(path[::-1][:lengths[b]])
        
        return batch_paths


class BiLSTMCRFModel(nn.Module):
    """BiLSTM + CRF model for sequence tagging."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_tags: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize BiLSTM+CRF model."""
        super(BiLSTMCRFModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRFLayer(num_tags)
        self.dropout = nn.Dropout(dropout)
        self.num_tags = num_tags

    def forward(self, x, lengths, tags=None):
        """
        Forward pass.

        Args:
            x: Character sequences
            lengths: Sequence lengths
            tags: Gold tags (for training)

        Returns:
            Loss (training) or predictions (inference)
        """
        embedded = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.fc(output)

        if tags is not None:
            # Training: return loss
            return self.crf(logits, tags, lengths)
        else:
            # Inference: return predictions
            return logits


class BiLSTMCRFTrainer:
    """Trainer for BiLSTM+CRF model."""

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

        self.model = BiLSTMCRFModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_tags=2,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.tag_to_idx = {"O": 0, "B": 1}
        self.idx_to_tag = {0: "O", 1: "B"}

    def _syllables_to_sequences(self, sentences):
        """Convert syllables to character sequences."""
        char_vocab = set()
        for sent in sentences:
            for syllable, _ in sent:
                char_vocab.update(syllable)

        char_vocab = sorted(list(char_vocab))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(char_vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        sequences = []
        labels = []

        for sent in sentences:
            seq = []
            tag_seq = []
            for syllable, tag in sent:
                char_seq = [self.char_to_idx.get(c, 0) for c in syllable]
                seq.extend(char_seq)
                tag_seq.extend([self.tag_to_idx[tag]] * len(char_seq))

            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(tag_seq, dtype=torch.long))

        return sequences, labels

    def train(self, sentences, epochs: int = 50, test_size: float = TEST_SPLIT_RATIO) -> Dict[str, Any]:
        """Train BiLSTM+CRF model."""
        logger.info("Preparing data for BiLSTM+CRF training...")

        sequences, labels = self._syllables_to_sequences(sentences)

        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=RANDOM_STATE
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        logger.info(f"Training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            batch_count = 0

            for i in range(0, len(X_train), self.batch_size):
                batch_seqs = X_train[i : i + self.batch_size]
                batch_labels = y_train[i : i + self.batch_size]

                lengths = torch.tensor([len(seq) for seq in batch_seqs])
                padded_seqs = pad_sequence(batch_seqs, batch_first=True)
                padded_labels = pad_sequence(batch_labels, batch_first=True)

                padded_seqs = padded_seqs.to(self.device)
                padded_labels = padded_labels.to(self.device)

                loss = self.model(padded_seqs, lengths.to(self.device), padded_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / batch_count:.4f}")

        # Evaluate
        logger.info("Evaluating BiLSTM+CRF model...")
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
                
                # Viterbi decoding from CRF
                preds = self.model.crf.decode(logits, lengths)

                for pred, label, length in zip(preds, padded_labels, lengths):
                    y_pred_flat.extend(pred)
                    y_test_flat.extend(label[:length].numpy())

        y_pred_flat = np.array(y_pred_flat)
        y_test_flat = np.array(y_test_flat)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_flat, y_pred_flat, average="weighted"
        )

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(np.mean(y_pred_flat == y_test_flat)),
        }

        logger.info(f"BiLSTM+CRF Metrics: {metrics}")

        return {
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict(self, sentences):
        """
        Predict tags for sentences.
        
        Args:
            sentences: List of syllable lists (each syllable is a string)
        
        Returns:
            List of tag predictions for each sentence
        """
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for sentence in sentences:
                # Convert syllables to character sequence (concatenated, like in training)
                seq = []
                syllable_boundaries = [0]  # Track where each syllable ends in the char sequence
                
                for syllable in sentence:
                    char_seq = [self.char_to_idx.get(c, 0) for c in syllable]
                    seq.extend(char_seq)
                    syllable_boundaries.append(len(seq))
                
                if not seq:
                    # Empty input, return default tags
                    predictions.append(["B"] * len(sentence))
                    continue
                
                # Create sequence tensor
                seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # Add batch dimension
                seq_tensor = seq_tensor.to(self.device)
                
                # Get lengths tensor (one length per sequence in batch)
                lengths = torch.tensor([len(seq)], dtype=torch.long).to(self.device)
                
                # Forward pass
                logits = self.model(seq_tensor, lengths)
                
                # Viterbi decoding
                pred_char_tags = self.model.crf.decode(logits, lengths)[0]
                
                # Convert character-level predictions back to syllable-level
                pred_tags = []
                for i in range(len(sentence)):
                    # Get the tag for the first character of this syllable
                    char_idx = syllable_boundaries[i]
                    if char_idx < len(pred_char_tags):
                        char_tag = pred_char_tags[char_idx]
                        tag_name = self.idx_to_tag.get(char_tag, "B")
                        pred_tags.append(tag_name)
                    else:
                        pred_tags.append("B")
                
                predictions.append(pred_tags)
        
        return predictions

