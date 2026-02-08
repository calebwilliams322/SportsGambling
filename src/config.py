"""
Central configuration for the SportsBetting project.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Seasons to pull ---
SEASONS = list(range(2016, 2026))

# --- Target columns (what we predict) ---
TARGET_STATS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "carries",
    "receptions",
    "receiving_yards",
    "receiving_tds",
]

# --- Rolling window sizes for feature engineering ---
ROLLING_WINDOWS = [3, 5, 8]

# --- Columns from weekly data used as rolling feature bases ---
ROLLING_STAT_COLUMNS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_epa",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
    "fantasy_points",
    "fantasy_points_ppr",
]

# --- Game context features (from schedules merge) ---
GAME_CONTEXT_FEATURES = [
    "is_home",
    "spread_line",
    "total_line",
    "is_dome",
    "temp",
    "wind",
]

# --- NGS feature columns (from Next Gen Stats merge) ---
NGS_PASSING_FEATURES = [
    "avg_time_to_throw",
    "avg_completed_air_yards",
    "avg_intended_air_yards",
    "aggressiveness",
    "avg_air_yards_to_sticks",
    "completion_percentage_above_expectation",
]

NGS_RUSHING_FEATURES = [
    "efficiency",
    "avg_time_to_los",
    "percent_attempts_gte_eight_defenders",
]

NGS_RECEIVING_FEATURES = [
    "avg_cushion",
    "avg_separation",
    "avg_yac_above_expectation",
]

# --- Walk-forward split ---
# Use 2024 as test (clean data from import_weekly_data).
# 2025 data (PBP fallback) is available for rolling features in predictions
# but has some missing columns so it's not ideal as a test set.
TRAIN_SEASONS = list(range(2016, 2023))   # 2016-2022
VAL_SEASONS = [2023]
TEST_SEASONS = [2024]

# --- Position filters per stat (only train on relevant positions) ---
STAT_POSITION_FILTER = {
    "passing_yards": ["QB"],
    "passing_tds": ["QB"],
    "rushing_yards": ["RB", "QB", "WR", "FB"],
    "carries": ["RB", "QB", "WR", "FB"],
    "receptions": ["WR", "TE", "RB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "receiving_tds": ["WR", "TE", "RB"],
}

# --- Model hyperparameters (MVP defaults) ---
MODEL_DEFAULTS = {
    "hidden_sizes": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10,  # early stopping patience
}
