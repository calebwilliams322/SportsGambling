"""
Feature engineering — all features are backward-looking only (no data leakage).

Creates rolling averages, opponent defensive stats, game context features,
and temporal features from the merged dataset.
"""
import pandas as pd
import numpy as np
from src.config import (
    DATA_PROCESSED, ROLLING_WINDOWS, ROLLING_STAT_COLUMNS,
    NGS_PASSING_FEATURES, NGS_RUSHING_FEATURES, NGS_RECEIVING_FEATURES,
)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling averages and standard deviations for player stats.

    Uses shift(1) so the current game is never included — only past games.
    Windows: 3, 5, 8 games.
    """
    # Sort by player and time
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Only compute rolling stats for columns that exist
    stat_cols = [c for c in ROLLING_STAT_COLUMNS if c in df.columns]

    for window in ROLLING_WINDOWS:
        print(f"  Computing {window}-game rolling features...")
        for col in stat_cols:
            grouped = df.groupby("player_id")[col]

            # Shift by 1 so we exclude the current game
            shifted = grouped.shift(1)

            # Rolling mean
            df[f"{col}_avg_{window}"] = shifted.rolling(
                window=window, min_periods=1
            ).mean()

            # Rolling standard deviation (consistency measure)
            df[f"{col}_std_{window}"] = shifted.rolling(
                window=window, min_periods=2
            ).std()

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend features — difference between short-term and long-term averages.
    Positive = player trending up. Negative = trending down.
    """
    stat_cols = [c for c in ROLLING_STAT_COLUMNS if c in df.columns]

    for col in stat_cols:
        short = f"{col}_avg_3"
        long = f"{col}_avg_8"
        if short in df.columns and long in df.columns:
            df[f"{col}_trend"] = df[short] - df[long]

    return df


def add_season_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season-to-date averages (expanding mean up to but not including current game).
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    stat_cols = [c for c in ROLLING_STAT_COLUMNS if c in df.columns]

    for col in stat_cols:
        shifted = df.groupby(["player_id", "season"])[col].shift(1)
        df[f"{col}_season_avg"] = shifted.expanding(min_periods=1).mean()

    return df


def add_opponent_defensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent defensive averages — how many stats does this defense allow
    to the position group?

    For each (opponent, season, week), compute the rolling average of stats
    allowed to each position (QB, RB, WR, TE) using only prior weeks.
    """
    if "opponent" not in df.columns or "position" not in df.columns:
        print("  Warning: Missing opponent/position columns, skipping opponent features.")
        return df

    # Key stats that defenses affect
    opp_stats = ["passing_yards", "rushing_yards", "receiving_yards",
                 "receptions", "passing_tds", "rushing_tds", "receiving_tds"]
    opp_stats = [c for c in opp_stats if c in df.columns]

    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    for stat in opp_stats:
        print(f"  Computing opponent defense avg for {stat}...")
        # Group by (opponent defense, position, season) and compute
        # expanding mean of stats they've allowed, shifted by 1 week
        grouped = df.groupby(["opponent", "position", "season"])[stat]
        shifted = grouped.shift(1)
        df[f"opp_def_{stat}_avg"] = shifted.expanding(min_periods=1).mean()

    return df


def add_ngs_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling averages for Next Gen Stats columns.
    These are already prefixed with ngs_ from the merge step.
    """
    ngs_cols = []
    for feat in NGS_PASSING_FEATURES:
        col = f"ngs_passing_{feat}"
        if col in df.columns:
            ngs_cols.append(col)
    for feat in NGS_RUSHING_FEATURES:
        col = f"ngs_rushing_{feat}"
        if col in df.columns:
            ngs_cols.append(col)
    for feat in NGS_RECEIVING_FEATURES:
        col = f"ngs_receiving_{feat}"
        if col in df.columns:
            ngs_cols.append(col)

    if not ngs_cols:
        print("  No NGS columns found, skipping NGS rolling features.")
        return df

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for col in ngs_cols:
        shifted = df.groupby("player_id")[col].shift(1)
        df[f"{col}_avg_3"] = shifted.rolling(window=3, min_periods=1).mean()
        df[f"{col}_avg_5"] = shifted.rolling(window=5, min_periods=1).mean()

    return df


def add_snap_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling average of snap share percentage."""
    if "offense_pct" not in df.columns:
        return df

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    shifted = df.groupby("player_id")["offense_pct"].shift(1)

    for window in ROLLING_WINDOWS:
        df[f"snap_pct_avg_{window}"] = shifted.rolling(
            window=window, min_periods=1
        ).mean()

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal/situational features:
    - week number
    - is_post_bye (did the team just have a bye week?)
    - days since last game (approximation based on week gaps)
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Week number as feature (normalized)
    df["week_num"] = df["week"]

    # Detect bye weeks: if a player's previous game was >1 week ago
    prev_week = df.groupby(["player_id", "season"])["week"].shift(1)
    df["weeks_since_last_game"] = df["week"] - prev_week
    df["is_post_bye"] = (df["weeks_since_last_game"] > 1).astype(int)

    # Fill NaN for first game of season
    df["weeks_since_last_game"] = df["weeks_since_last_game"].fillna(1)

    return df


def add_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up game context features from the schedules merge.
    Fill missing weather data for dome games.
    """
    # Fill missing temp/wind for dome games
    if "is_dome" in df.columns and "temp" in df.columns:
        df.loc[df["is_dome"] == 1, "temp"] = df.loc[df["is_dome"] == 1, "temp"].fillna(72)
        df.loc[df["is_dome"] == 1, "wind"] = df.loc[df["is_dome"] == 1, "wind"].fillna(0)

    # Fill remaining missing weather with median
    if "temp" in df.columns:
        df["temp"] = df["temp"].fillna(df["temp"].median())
    if "wind" in df.columns:
        df["wind"] = df["wind"].fillna(df["wind"].median())

    return df


def build_features() -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Loads merged data, adds all features, saves result.
    """
    print("Loading merged dataset...")
    df = pd.read_parquet(DATA_PROCESSED / "merged_player_weekly.parquet")
    print(f"  Shape: {df.shape}")

    print("\nAdding rolling player averages (3/5/8 game windows)...")
    df = add_rolling_features(df)

    print("Adding trend features...")
    df = add_trend_features(df)

    print("Adding season-to-date averages...")
    df = add_season_averages(df)

    print("Adding opponent defensive features...")
    df = add_opponent_defensive_features(df)

    print("Adding NGS rolling features...")
    df = add_ngs_rolling_features(df)

    print("Adding snap share rolling features...")
    df = add_snap_rolling_features(df)

    print("Adding temporal features...")
    df = add_temporal_features(df)

    print("Cleaning game context features...")
    df = add_game_context_features(df)

    # Save
    out_path = DATA_PROCESSED / "features_player_weekly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved feature dataset: {out_path}")
    print(f"Final shape: {df.shape}")

    return df
