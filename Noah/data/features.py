import pandas as pd
import numpy as np

from config import ROLLING_WINDOW, MIN_PERIODS, PROP_TYPES
from data.fetch import fetch_weekly_stats, fetch_schedules


def compute_player_rolling(df, stat_cols, window=ROLLING_WINDOW, min_periods=MIN_PERIODS):
    """Compute rolling averages of stat_cols per player, shifted by 1 to prevent leakage."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    for col in stat_cols:
        df[f"{col}_roll{window}"] = (
            df.groupby("player_id")[col]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean())
        )
    return df


def compute_opponent_defensive_rolling(weekly_df, stat_col, window=ROLLING_WINDOW, min_periods=MIN_PERIODS):
    """Compute rolling average of total stat_col allowed by each defense."""
    # Sum stat per game for the defense
    def_game = (
        weekly_df.groupby(["opponent_team", "season", "week"])[stat_col]
                 .sum()
                 .reset_index()
                 .rename(columns={stat_col: f"def_{stat_col}_total"})
    )
    def_game = def_game.sort_values(["opponent_team", "season", "week"])

    def_game[f"opp_{stat_col}_allowed_roll{window}"] = (
        def_game.groupby("opponent_team")[f"def_{stat_col}_total"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean())
    )
    return def_game[["opponent_team", "season", "week", f"opp_{stat_col}_allowed_roll{window}"]]


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
    Build the full feature matrix for a given prop type.
    Returns a DataFrame with feature columns, target column, and metadata columns.
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

    # Player rolling averages
    df = compute_player_rolling(df, cfg["player_features"])

    # Opponent defensive rolling averages
    def_rolling = compute_opponent_defensive_rolling(weekly, cfg["defensive_stat"])
    df = df.merge(def_rolling, on=["opponent_team", "season", "week"], how="left")

    # Home/away
    df = add_home_away(df, schedules)

    # Rest days
    df = add_rest_days(df, schedules)

    # Postseason flag
    df["is_postseason"] = (df["season_type"] == "POST").astype(int)

    # Build feature column list
    roll_cols = [f"{c}_roll{ROLLING_WINDOW}" for c in cfg["player_features"]]
    opp_col = f"opp_{cfg['defensive_stat']}_allowed_roll{ROLLING_WINDOW}"
    feature_cols = roll_cols + [opp_col, "is_home", "rest_days", "week", "is_postseason"]

    # Keep only rows with all features present
    keep_cols = feature_cols + [prop_type, "player_id", "player_display_name", "season", "recent_team", "opponent_team"]
    df = df[keep_cols].dropna(subset=feature_cols)

    return df, feature_cols
