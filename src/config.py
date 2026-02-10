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

# --- Chronological train/val/test split ---
# Train: all data up to 2025 week 14 (includes all active players)
# Val:   2025 weeks 15-18 (late regular season, for early stopping)
# Test:  2025 weeks 19+ (playoffs — closest to Super Bowl predictions)
TRAIN_CUTOFF = (2025, 14)   # (season, last_week) included in training
VAL_CUTOFF = (2025, 18)     # (season, last_week) included in validation

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

# --- Features to exclude (unavailable in 2025 PBP data) ---
# These are derived from snap counts and Next Gen Stats, which aren't
# available for the current season via PBP fallback. Including them
# would create a train/predict mismatch (real values in training, 0s at prediction).
EXCLUDED_FEATURES = [
    "offense_snaps",
    "offense_pct",
    "snap_pct_avg_3",
    "snap_pct_avg_5",
    "snap_pct_avg_8",
    "ngs_passing_avg_time_to_throw_avg_3",
    "ngs_passing_avg_time_to_throw_avg_5",
    "ngs_passing_avg_completed_air_yards_avg_3",
    "ngs_passing_avg_completed_air_yards_avg_5",
    "ngs_passing_avg_intended_air_yards_avg_3",
    "ngs_passing_avg_intended_air_yards_avg_5",
    "ngs_passing_aggressiveness_avg_3",
    "ngs_passing_aggressiveness_avg_5",
    "ngs_passing_avg_air_yards_to_sticks_avg_3",
    "ngs_passing_avg_air_yards_to_sticks_avg_5",
    "ngs_passing_completion_percentage_above_expectation_avg_3",
    "ngs_passing_completion_percentage_above_expectation_avg_5",
    "ngs_rushing_efficiency_avg_3",
    "ngs_rushing_efficiency_avg_5",
    "ngs_rushing_avg_time_to_los_avg_3",
    "ngs_rushing_avg_time_to_los_avg_5",
    "ngs_rushing_percent_attempts_gte_eight_defenders_avg_3",
    "ngs_rushing_percent_attempts_gte_eight_defenders_avg_5",
    "ngs_receiving_avg_cushion_avg_3",
    "ngs_receiving_avg_cushion_avg_5",
    "ngs_receiving_avg_separation_avg_3",
    "ngs_receiving_avg_separation_avg_5",
    "ngs_receiving_avg_yac_above_expectation_avg_3",
    "ngs_receiving_avg_yac_above_expectation_avg_5",
]

# --- Model hyperparameters (MVP defaults) ---
MODEL_DEFAULTS = {
    "hidden_sizes": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10,  # early stopping patience
}
