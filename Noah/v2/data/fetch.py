"""Thin wrapper reusing v1's data fetching functions."""

import sys
import os

# Add project root to path so we can import v1 code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.fetch import fetch_weekly_stats, fetch_schedules

__all__ = ["fetch_weekly_stats", "fetch_schedules"]
