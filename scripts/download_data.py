#!/usr/bin/env python3
"""
Download all NFL data from nfl_data_py and cache locally as parquet files.
Run this first before any other scripts.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest import download_all
from src.config import SEASONS

if __name__ == "__main__":
    print(f"Downloading NFL data for seasons {SEASONS[0]}-{SEASONS[-1]}...")
    print("This may take a few minutes on first run.\n")
    download_all(SEASONS)
