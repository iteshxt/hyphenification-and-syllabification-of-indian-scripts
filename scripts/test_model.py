#!/usr/bin/env python3
"""
Interactive Model Testing
Input: Devanagari words
Output: Segmented syllables with confidence scores
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import SyllableSegmenter
from src.config import MODEL_PATH

# Load model
segmenter = SyllableSegmenter(MODEL_PATH)

print("\n" + "="*50)
print("SYLLABLE SEGMENTATION")
print("="*50)
print("Enter Devanagari words (or 'quit' to exit)\n")

while True:
    word = input("Enter word: ").strip()
    
    if word.lower() == 'quit':
        print("\nExit.\n")
        break
    
    if not word:
        print("⚠️  Please enter a word\n")
        continue
    
    try:
        # Get syllables
        syllables = segmenter.segment_word(word)
        
        # Get with confidence
        results = segmenter.segment_word_with_confidence(word)
        
        # Display output
        print(f"\nInput:  {word}")
        print(f"Output: {' + '.join(syllables)}")
        print("\nDetailed:")
        for item in results:
            confidence = item.get('confidence', 0)
            syllable = item.get('syllable', '')
            bar = '█' * int(confidence * 10) + '░' * (10 - int(confidence * 10))
            print(f"  {syllable:10} [{bar}] {confidence:.2%}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
