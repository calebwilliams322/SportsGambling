"""
Expanded feature engineering for v2.

Multi-window rolling averages (3/5/10 games), player consistency (std dev),
efficiency metrics, trend (slope), and position-specific defensive stats.
"""

import pandas as pd
import numpy as np

from v2.config import ROLLING_WINDOWS, MIN_PERIODS, PROP_TYPES
from v2.data.fetch import fetch_weekly_stats, fetch_schedules


def compute_player_rolling_multi(df, stat_cols, windows=ROLLING_WINDOWS, min_periods=MIN_PERIODS):
    """Compute rolling mean AND std for multiple windows, shifted by 1 to prevent leakage."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    for col in stat_cols:
        shifted = df.groupby("player_id")[col].transform(lambda x: x.shift(1))
        for w in windows:
            df[f"{col}_roll{w}"] = (
                shifted.groupby(df["player_id"])
                       .transform(lambda x: x.rolling(w, min_periods=min_periods).mean())
            )
        # Std dev over largest window (player consistency)
        max_w = max(windows)
        df[f"{col}_std{max_w}"] = (
            shifted.groupby(df["player_id"])
                   .transform(lambda x: x.rolling(max_w, min_periods=min_periods).std())
        )
    return df


def compute_expanding_season_avg(df, stat_cols):
    """Season-long expanding average per player (shifted by 1)."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    for col in stat_cols:
        df[f"{col}_season_avg"] = (
            df.groupby(["player_id", "season"])[col]
              .transform(lambda x: x.shift(1).expanding(min_periods=MIN_PERIODS).mean())
        )
    return df


def compute_efficiency_features(df, efficiency_map):
    """Compute ratio features like yards_per_target, completion_rate using rolling data."""
    max_w = max(ROLLING_WINDOWS)
    for name, (numerator, denominator) in efficiency_map.items():
        num_col = f"{numerator}_roll{max_w}"
        den_col = f"{denominator}_roll{max_w}"
        if num_col in df.columns and den_col in df.columns:
            df[name] = df[num_col] / df[den_col].replace(0, np.nan)
    return df


def compute_player_trend(df, stat_col, window=5):
    """Linear slope of the main stat over the last `window` games (momentum/trend)."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def slope(vals):
        if len(vals) < window:
            return np.nan
        y = vals.values[-window:]
        if len(y) < window:
            return np.nan
        y_mean = y.mean()
        return ((x * (y - y_mean)).sum()) / x_var if x_var > 0 else 0.0

    df[f"{stat_col}_trend"] = (
        df.groupby("player_id")[stat_col]
          .transform(lambda s: s.shift(1).rolling(window, min_periods=window).apply(slope, raw=False))
    )
    return df


def compute_opponent_defensive_rolling(weekly_df, stat_col, windows=ROLLING_WINDOWS, min_periods=MIN_PERIODS):
    """Rolling average of stat allowed by each defense, for multiple windows."""
    def_game = (
        weekly_df.groupby(["opponent_team", "season", "week"])[stat_col]
                 .sum()
                 .reset_index()
                 .rename(columns={stat_col: f"def_{stat_col}_total"})
    )
    def_game = def_game.sort_values(["opponent_team", "season", "week"])

    shifted = def_game.groupby("opponent_team")[f"def_{stat_col}_total"].transform(lambda x: x.shift(1))
    for w in windows:
        def_game[f"opp_{stat_col}_allowed_roll{w}"] = (
            shifted.groupby(def_game["opponent_team"])
                   .transform(lambda x: x.rolling(w, min_periods=min_periods).mean())
        )

    result_cols = ["opponent_team", "season", "week"]
    result_cols += [f"opp_{stat_col}_allowed_roll{w}" for w in windows]
    return def_game[result_cols]


def compute_position_defensive_stats(weekly_df, stat_col, positions, windows=ROLLING_WINDOWS, min_periods=MIN_PERIODS):
    """Opponent defense stats broken down by position (e.g., yards allowed to WRs vs TEs)."""
    result_frames = []
    for pos in positions:
        pos_df = weekly_df[weekly_df["position"] == pos].copy()
        def_game = (
            pos_df.groupby(["opponent_team", "season", "week"])[stat_col]
                  .sum()
                  .reset_index()
                  .rename(columns={stat_col: f"def_{stat_col}_{pos}_total"})
        )
        def_game = def_game.sort_values(["opponent_team", "season", "week"])

        max_w = max(windows)
        col_name = f"def_{stat_col}_{pos}_total"
        shifted = def_game.groupby("opponent_team")[col_name].transform(lambda x: x.shift(1))
        def_game[f"opp_{stat_col}_to_{pos}_roll{max_w}"] = (
            shifted.groupby(def_game["opponent_team"])
                   .transform(lambda x: x.rolling(max_w, min_periods=min_periods).mean())
        )
        result_frames.append(
            def_game[["opponent_team", "season", "week", f"opp_{stat_col}_to_{pos}_roll{max_w}"]]
        )

    if not result_frames:
        return pd.DataFrame()

    result = result_frames[0]
    for frame in result_frames[1:]:
        result = result.merge(frame, on=["opponent_team", "season", "week"], how="outer")
    return result


def add_home_away(player_df, schedules_df):
    """Add is_home binary feature."""
    home = schedules_df[["season", "week", "home_team"]].copy()
    home["is_home"] = 1
    home = home.rename(columns={"home_team": "team"})

    away = schedules_df[["season", "week", "away_team"]].copy()
    away["is_home"] = 0
    away = away.rename(columns={"away_team": "team"})

    lookup = pd.concat([home, away], ignore_index=True)
    return player_df.merge(
        lookup,
        left_on=["recent_team", "season", "week"],
        right_on=["team", "season", "week"],
        how="left",
    ).drop(columns=["team"])


def add_rest_days(player_df, schedules_df):
    """Add rest_days feature from schedule data."""
    home = schedules_df[["season", "week", "home_team", "home_rest"]].rename(
        columns={"home_team": "team", "home_rest": "rest_days"}
    )
    away = schedules_df[["season", "week", "away_team", "away_rest"]].rename(
        columns={"away_team": "team", "away_rest": "rest_days"}
    )
    lookup = pd.concat([home, away], ignore_index=True)
    return player_df.merge(
        lookup,
        left_on=["recent_team", "season", "week"],
        right_on=["team", "season", "week"],
        how="left",
    ).drop(columns=["team"])


def build_features(prop_type):
    """
    Build the expanded feature matrix for a given prop type.
    Returns (df, feature_cols) — DataFrame with features, target, and metadata.
    """
    cfg = PROP_TYPES[prop_type]
    weekly = fetch_weekly_stats()
    schedules = fetch_schedules()

    # Filter to relevant positions and minimum activity
    mask = (
        weekly["position"].isin(cfg["position_filter"])
        & (weekly[cfg["min_filter_col"]] >= cfg["min_filter_val"])
    )
    df = weekly[mask].copy()

    # ── Player rolling averages (multi-window) ────────────────────────
    df = compute_player_rolling_multi(df, cfg["player_features"])

    # ── Season-long expanding averages ────────────────────────────────
    main_stat = cfg["player_features"][0]  # e.g., "passing_yards"
    df = compute_expanding_season_avg(df, [main_stat])

    # ── Efficiency features ───────────────────────────────────────────
    df = compute_efficiency_features(df, cfg.get("efficiency_features", {}))

    # ── Player trend (slope) ──────────────────────────────────────────
    df = compute_player_trend(df, main_stat)

    # ── Opponent defensive rolling (multi-window) ─────────────────────
    def_rolling = compute_opponent_defensive_rolling(weekly, cfg["defensive_stat"])
    df = df.merge(def_rolling, on=["opponent_team", "season", "week"], how="left")

    # ── Position-specific defensive stats ─────────────────────────────
    pos_def = compute_position_defensive_stats(
        weekly, cfg["defensive_stat"], cfg["position_filter"]
    )
    if not pos_def.empty:
        df = df.merge(pos_def, on=["opponent_team", "season", "week"], how="left")

    # ── Home/away ─────────────────────────────────────────────────────
    df = add_home_away(df, schedules)

    # ── Rest days ─────────────────────────────────────────────────────
    df = add_rest_days(df, schedules)

    # ── Postseason flag ───────────────────────────────────────────────
    df["is_postseason"] = (df["season_type"] == "POST").astype(int)

    # ── Build feature column list ─────────────────────────────────────
    feature_cols = []

    # Multi-window rolling means
    for col in cfg["player_features"]:
        for w in ROLLING_WINDOWS:
            feature_cols.append(f"{col}_roll{w}")
        feature_cols.append(f"{col}_std{max(ROLLING_WINDOWS)}")

    # Season expanding average
    feature_cols.append(f"{main_stat}_season_avg")

    # Efficiency features
    for name in cfg.get("efficiency_features", {}).keys():
        feature_cols.append(name)

    # Trend
    feature_cols.append(f"{main_stat}_trend")

    # Opponent defensive (multi-window)
    for w in ROLLING_WINDOWS:
        feature_cols.append(f"opp_{cfg['defensive_stat']}_allowed_roll{w}")

    # Position-specific defense
    max_w = max(ROLLING_WINDOWS)
    for pos in cfg["position_filter"]:
        col = f"opp_{cfg['defensive_stat']}_to_{pos}_roll{max_w}"
        if col in df.columns:
            feature_cols.append(col)

    # Context features
    feature_cols += ["is_home", "rest_days", "week", "is_postseason"]

    # Remove duplicate columns from merges
    df = df.loc[:, ~df.columns.duplicated()]

    # Keep only rows with all features present
    metadata_cols = ["player_id", "player_display_name", "season",
                     "recent_team", "opponent_team", "position"]
    # 'week' is already in feature_cols, so don't add it again via metadata
    keep_cols = feature_cols + [prop_type] + [c for c in metadata_cols if c in df.columns and c not in feature_cols]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df = df.dropna(subset=[c for c in feature_cols if c in df.columns])

    # Filter to only feature columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols
