import os

# Data
YEARS = list(range(2015, 2026))
ROLLING_WINDOW = 5
MIN_PERIODS = 3

# Model
HIDDEN_DIMS = [64, 32]
DROPOUT = 0.2

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATIENCE = 10

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cache")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Train/Val/Test split by season
TRAIN_SEASONS = list(range(2015, 2023))   # 2015-2022
VAL_SEASONS = [2023, 2024]
TEST_SEASONS = [2025]

# Prop type configurations
PROP_TYPES = {
    "passing_yards": {
        "position_filter": ["QB"],
        "min_filter_col": "attempts",
        "min_filter_val": 1,
        "player_features": [
            "passing_yards", "attempts", "completions", "passing_tds",
            "interceptions", "passing_air_yards", "passing_yards_after_catch",
            "passing_epa",
        ],
        "defensive_stat": "passing_yards",
    },
    "rushing_yards": {
        "position_filter": ["QB", "RB", "WR"],
        "min_filter_col": "carries",
        "min_filter_val": 1,
        "player_features": [
            "rushing_yards", "carries", "rushing_tds",
            "rushing_first_downs", "rushing_epa",
        ],
        "defensive_stat": "rushing_yards",
    },
    "receiving_yards": {
        "position_filter": ["WR", "TE", "RB"],
        "min_filter_col": "targets",
        "min_filter_val": 1,
        "player_features": [
            "receiving_yards", "targets", "receptions", "receiving_tds",
            "receiving_air_yards", "receiving_yards_after_catch",
            "receiving_epa", "target_share",
        ],
        "defensive_stat": "receiving_yards",
    },
}
