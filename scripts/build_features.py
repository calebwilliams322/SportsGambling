#!/usr/bin/env python3
"""
Merge all raw datasets and engineer features.
Run after download_data.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.merge import build_merged_dataset
from src.data.features import build_features

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Merging raw datasets")
    print("=" * 60)
    build_merged_dataset()

    print("\n" + "=" * 60)
    print("STEP 2: Engineering features")
    print("=" * 60)
    build_features()

    print("\nDone! Features ready for training.")
