"""
Utility Helper Functions for COMP5423 RAG System.

This module provides utility functions for data conversion and file operations.
It includes functions for safely handling numpy types and saving prediction results.

Key Features:
- Safe conversion of numpy types to Python native types
- JSON serialization with proper encoding
- Prediction file saving with data cleaning

Important Notes:
- Handles numpy type conversion for JSON serialization
- Ensures proper Unicode encoding for Chinese text
- Provides clean data formatting for output files
"""

import json
import numpy as np


def safe_convert(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar → python scalar

    if isinstance(obj, (list, tuple)):
        return [safe_convert(v) for v in obj]

    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}

    return obj


def save_predictions(results, file_path):
    """Save prediction results to JSONL file with proper type conversion and encoding."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in results:
            cleaned = safe_convert(item)  # Main fix
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    print("save predictions done.")