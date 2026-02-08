import os

# ── Data ──────────────────────────────────────────────────────────────
YEARS = list(range(2015, 2026))
ROLLING_WINDOWS = [3, 5, 10]
MIN_PERIODS = 3

# ── Neural Network ────────────────────────────────────────────────────
NN_HIDDEN_DIMS = [128, 64, 32]
NN_DROPOUT = 0.3
NN_LEARNING_RATE = 1e-3
NN_WEIGHT_DECAY = 1e-5
NN_BATCH_SIZE = 64
NN_EPOCHS = 150
NN_PATIENCE = 15

# ── LightGBM ─────────────────────────────────────────────────────────
GBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
}

# ── Ensemble ──────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {"nn": 0.4, "gbm": 0.6}

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
V1_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

# ── Train / Val / Test split by season ────────────────────────────────
TRAIN_SEASONS = list(range(2015, 2024))   # 2015-2023
VAL_SEASONS = [2024]                       # optimize ensemble weights
TEST_SEASONS = [2025]                      # final evaluation
CV_SEASONS = list(range(2020, 2026))       # leave-one-season-out CV

# ── Prop type configurations ─────────────────────────────────────────
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
        "efficiency_features": {
            "completion_rate": ("completions", "attempts"),
        },
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
        "efficiency_features": {
            "yards_per_carry": ("rushing_yards", "carries"),
        },
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
        "efficiency_features": {
            "yards_per_target": ("receiving_yards", "targets"),
            "catch_rate": ("receptions", "targets"),
        },
    },
}
