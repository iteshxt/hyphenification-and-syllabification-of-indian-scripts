#!/usr/bin/env python3
"""
Interactive Devanagari syllabification.
Run and enter words to get syllable breakdown.

Usage:
    python main.py              # Use BiLSTM+CRF model (default)
    python main.py --crf        # Use CRF model
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import SyllableSegmenter
from src.config import MODEL_PATH, BILSTM_CRF_MODEL_PATH

# Parse arguments
use_crf = "--crf" in sys.argv
model_path = MODEL_PATH if use_crf else BILSTM_CRF_MODEL_PATH
model_type = "CRF" if use_crf else "BiLSTM+CRF"

# Check if model exists
if not Path(model_path).exists():
    print(f"❌ {model_type} model not found at {model_path}")
    if use_crf:
        print("Please run: python scripts/train.py")
    else:
        print("Please run: python scripts/train_bilstm_crf.py")
    sys.exit(1)

# Load model
print(f"📦 Loading {model_type} model...")
try:
    segmenter = SyllableSegmenter(model_path)
    print(f"✓ {model_type} model loaded!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    if use_crf:
        print("Make sure you've trained the model: python scripts/train.py")
    else:
        print("Make sure you've trained the model: python scripts/train_bilstm_crf.py")
    sys.exit(1)

print("=" * 60)
print("DEVANAGARI SYLLABIFICATION - INTERACTIVE MODE")
print("=" * 60)
print("Enter Devanagari words to segment into syllables.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        # Get input
        word = input("Enter word: ").strip()
        
        if word.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not word:
            print("⚠️  Please enter a word.\n")
            continue
        
        # Process
        syllables = segmenter.segment_word(word)
        
        # Display results
        print(f"\n  Input:        {word}")
        print(f"  Syllables:    {syllables}")
        print(f"  Hyphenated:   {'-'.join(syllables)}")
        print(f"  Count:        {len(syllables)}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye! 👋")
        break
    except Exception as e:
        print(f"❌ Error: {e}\n")
