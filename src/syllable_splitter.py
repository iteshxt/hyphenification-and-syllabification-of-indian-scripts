#!/usr/bin/env python3
"""
Devanagari Syllable Splitter
Splits words into syllable chunks using training data + linguistic rules.

Strategy:
1. First try to find exact match in training dataset
2. If not found, use linguistic rules to estimate syllables
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEVANAGARI_CONSONANTS, DEVANAGARI_VOWEL_SIGNS, DEVANAGARI_VIRAMA, RAW_DATASET

class SyllableSplitter:
    """Split Devanagari words into linguistic syllables."""
    
    # Class variable to cache training data
    _syllable_map: Optional[Dict[str, List[str]]] = None
    
    @classmethod
    def _load_training_data(cls) -> Dict[str, List[str]]:
        """Load syllable splits from training dataset."""
        if cls._syllable_map is not None:
            return cls._syllable_map
        
        cls._syllable_map = {}
        try:
            with open(RAW_DATASET, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    word = entry.get("word", "")
                    split = entry.get("split", [])
                    if word:
                        cls._syllable_map[word] = split
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
        
        return cls._syllable_map
    
    @staticmethod
    def _split_linguistic(word: str) -> List[str]:
        """
        Split using linguistic rules (fallback if not in training data).
        
        Rule: Consonant + optional(virama+consonant) + optional(vowel mark) = syllable
        """
        syllables = []
        i = 0
        
        while i < len(word):
            syllable = ""
            char = word[i]
            
            # Independent vowel
            if char not in DEVANAGARI_CONSONANTS and char != DEVANAGARI_VIRAMA and char not in DEVANAGARI_VOWEL_SIGNS:
                syllable = char
                i += 1
            
            # Consonant
            elif char in DEVANAGARI_CONSONANTS:
                syllable = char
                i += 1
                
                # Virama + consonant pairs
                while i < len(word) and word[i] == DEVANAGARI_VIRAMA:
                    if i + 1 < len(word) and word[i + 1] in DEVANAGARI_CONSONANTS:
                        syllable += word[i]  # virama
                        i += 1
                        syllable += word[i]  # consonant
                        i += 1
                    else:
                        syllable += word[i]
                        i += 1
                        break
                
                # Vowel mark
                if i < len(word) and word[i] in DEVANAGARI_VOWEL_SIGNS:
                    syllable += word[i]
                    i += 1
            
            # Vowel mark
            elif char in DEVANAGARI_VOWEL_SIGNS:
                syllable = char
                i += 1
            
            # Virama
            elif char == DEVANAGARI_VIRAMA:
                syllable = char
                i += 1
            
            if syllable:
                syllables.append(syllable)
        
        return syllables
    
    @classmethod
    def split(cls, word: str) -> List[str]:
        """
        Split a Devanagari word into syllables.
        
        Strategy:
        1. Check training data first
        2. Fall back to linguistic rules
        
        Args:
            word: Devanagari word
            
        Returns:
            List of syllables
            
        Examples:
            "कर्म" → ["कर्", "म"]
            "विद्यालय" → ["वि", "द्या", "लय"]
        """
        # Try training data first
        training_data = cls._load_training_data()
        if word in training_data:
            return training_data[word]
        
        # Fall back to linguistic rules
        return cls._split_linguistic(word)


# Test the splitter
if __name__ == "__main__":
    test_words = [
        "कर्म",
        "विद्यालय",
        "प्रार्थना",
        "संस्कृत",
        "अद्भुत",
    ]
    
    splitter = SyllableSplitter()
    for word in test_words:
        result = splitter.split(word)
        print(f"{word:15} → {' + '.join(result)}")
