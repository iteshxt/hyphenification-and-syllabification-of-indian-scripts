#!/usr/bin/env python3
"""
Simple Model Performance Check
Input: None (automatic)
Output: Model metrics
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_METRICS_PATH

# Load and display metrics
if Path(MODEL_METRICS_PATH).exists():
    with open(MODEL_METRICS_PATH, "r") as f:
        results = json.load(f)
    
    metrics = results.get("metrics", {})
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"F1-Score:  {metrics.get('f1', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Train Set: {results.get('train_size', 0)} sentences")
    print(f"Test Set:  {results.get('test_size', 0)} sentences")
    print(f"Features:  {results.get('feature_count', 0)}")
    print("="*50 + "\n")
else:
    print("\n❌ No metrics file found. Train model first:\n")
    print("$ python3 scripts/train.py\n")
