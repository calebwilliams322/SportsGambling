"""
PyTorch Dataset and DataLoader setup for player props prediction.
Handles train/val/test splitting by season (walk-forward).
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.config import (
    DATA_PROCESSED, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    MODEL_DEFAULTS, STAT_POSITION_FILTER,
)


class PlayerPropsDataset(Dataset):
    """
    PyTorch Dataset for a single target stat.

    Args:
        features: numpy array of shape (n_samples, n_features)
        targets: numpy array of shape (n_samples,)
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def get_feature_columns(df: pd.DataFrame, target_stat: str) -> list[str]:
    """
    Determine which columns to use as input features.
    Excludes identifiers, raw stat columns (to prevent leakage), and the target.
    """
    # Columns to exclude (identifiers, raw current-game stats, targets)
    exclude_prefixes = ["player_id", "player_name", "player_display_name",
                        "headshot_url", "game_id", "recent_team", "opponent",
                        "season_type", "report_status", "practice_status",
                        "roof", "surface"]

    # Raw current-game stat columns (these ARE the targets, can't use as features)
    raw_stats = [
        "completions", "attempts", "passing_yards", "passing_tds",
        "interceptions", "sacks", "sack_yards", "sack_fumbles",
        "sack_fumbles_lost", "passing_air_yards", "passing_yards_after_catch",
        "passing_first_downs", "passing_epa", "passing_2pt_conversions",
        "dakota", "pacr",
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
        "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
        "rushing_2pt_conversions",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
        "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
        "receiving_2pt_conversions", "racr", "target_share", "air_yards_share",
        "wopr",
        "special_teams_tds", "fantasy_points", "fantasy_points_ppr",
    ]

    # Also exclude raw NGS columns (use rolling versions instead)
    raw_ngs_prefixes = ["ngs_passing_", "ngs_rushing_", "ngs_receiving_"]

    feature_cols = []
    for col in df.columns:
        # Skip excluded prefixes
        if any(col.startswith(p) or col == p for p in exclude_prefixes):
            continue
        # Skip raw current-game stats
        if col in raw_stats:
            continue
        # Skip raw NGS columns (but keep rolling versions like ngs_passing_*_avg_3)
        if any(col.startswith(p) for p in raw_ngs_prefixes):
            if not any(col.endswith(f"_avg_{w}") for w in [3, 5]):
                continue
        # Skip non-numeric columns
        if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32, float, int]:
            continue
        feature_cols.append(col)

    return feature_cols


def prepare_data(target_stat: str):
    """
    Load features, split by season, normalize, and return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, scaler, feature_columns
    """
    df = pd.read_parquet(DATA_PROCESSED / "features_player_weekly.parquet")

    # Filter to players who actually have data for this stat
    df = df[df[target_stat].notna()].copy()

    # Position filtering — only train on relevant positions per stat
    positions = STAT_POSITION_FILTER.get(target_stat)
    if positions and "position" in df.columns:
        before = len(df)
        df = df[df["position"].isin(positions)].copy()
        print(f"Position filter ({positions}): {before} → {len(df)} rows")

    # Get feature columns
    feature_cols = get_feature_columns(df, target_stat)
    print(f"Using {len(feature_cols)} features for {target_stat}")

    # Split by season (walk-forward)
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    print(f"  Train: {len(train_df)} rows ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]})")
    print(f"  Val:   {len(val_df)} rows ({VAL_SEASONS})")
    print(f"  Test:  {len(test_df)} rows ({TEST_SEASONS})")

    # Drop rows where we don't have enough history (first few games per player)
    # Use the 3-game rolling avg as a proxy — if it's NaN, we don't have enough data
    history_col = f"{target_stat}_avg_3"
    if history_col in train_df.columns:
        train_df = train_df[train_df[history_col].notna()]
        val_df = val_df[val_df[history_col].notna()]
        test_df = test_df[test_df[history_col].notna()]
        print(f"  After history filter: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Extract features and targets
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_stat].values.astype(np.float32)

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_stat].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_stat].values.astype(np.float32)

    # Fill NaNs with 0 (missing data from left joins)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Normalize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Normalize targets (fit on train only) — critical for Huber loss to work
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    if target_std < 1e-6:
        target_std = 1.0  # avoid division by zero
    print(f"  Target normalization: mean={target_mean:.2f}, std={target_std:.2f}")

    y_train_norm = (y_train - target_mean) / target_std
    y_val_norm = (y_val - target_mean) / target_std
    y_test_norm = (y_test - target_mean) / target_std

    # Create datasets and loaders
    batch_size = MODEL_DEFAULTS["batch_size"]

    train_dataset = PlayerPropsDataset(X_train, y_train_norm)
    val_dataset = PlayerPropsDataset(X_val, y_val_norm)
    test_dataset = PlayerPropsDataset(X_test, y_test_norm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    target_stats_info = {"mean": target_mean, "std": target_std}
    return train_loader, val_loader, test_loader, scaler, feature_cols, target_stats_info
