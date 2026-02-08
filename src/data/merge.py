"""
Data merging — joins all raw datasets into a single master DataFrame.

Merge order:
1. weekly (base) — player stats per game
2. + schedules — game context (home/away, weather, spread, total)
3. + snap_counts — snap share per player per week
4. + ngs_passing/rushing/receiving — Next Gen Stats tracking metrics
5. + injuries — pre-game injury status
"""
import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_PROCESSED


def load_raw(filename: str) -> pd.DataFrame:
    """Load a parquet file from data/raw/."""
    return pd.read_parquet(DATA_RAW / filename)


def merge_schedules(weekly: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Merge game context onto player weekly data.

    Adds: is_home, spread_line, total_line, temp, wind, roof, surface,
          opponent team, game_id.

    Join logic: match player's team (recent_team) to either home_team or
    away_team in the schedule for that season/week.
    """
    # Keep only the columns we need from schedules
    sched_cols = [
        "season", "week", "game_type", "home_team", "away_team",
        "spread_line", "total_line", "temp", "wind", "roof", "surface",
        "home_score", "away_score", "game_id",
    ]
    sched = schedules[[c for c in sched_cols if c in schedules.columns]].copy()

    # Build a lookup: for each (season, week, team), get game context
    # Home teams
    home = sched.copy()
    home["team"] = home["home_team"]
    home["is_home"] = 1
    home["opponent"] = home["away_team"]
    # Spread is typically from home perspective, flip for away
    home["team_spread"] = home["spread_line"]

    # Away teams
    away = sched.copy()
    away["team"] = away["away_team"]
    away["is_home"] = 0
    away["opponent"] = away["home_team"]
    away["team_spread"] = -away["spread_line"]

    # Combine home and away views
    game_context = pd.concat([home, away], ignore_index=True)

    # Merge onto weekly data
    merged = weekly.merge(
        game_context[["season", "week", "team", "is_home", "opponent",
                       "team_spread", "total_line", "temp", "wind", "roof",
                       "surface", "game_id"]],
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"],
        how="left",
    )
    merged.drop(columns=["team"], inplace=True, errors="ignore")

    # Encode dome as binary
    if "roof" in merged.columns:
        merged["is_dome"] = merged["roof"].isin(["dome", "closed"]).astype(int)

    return merged


def merge_snap_counts(merged: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """
    Merge snap count data onto the master DataFrame.

    Adds: offense_pct, offense_snaps.

    Join logic: snap counts use player name + team + season + week.
    """
    snap_cols = ["player", "team", "season", "week", "offense_snaps", "offense_pct"]
    snap_subset = snaps[[c for c in snap_cols if c in snaps.columns]].copy()

    # Standardize name column for matching
    if "player" in snap_subset.columns:
        snap_subset.rename(columns={"player": "snap_player_name"}, inplace=True)

    merged = merged.merge(
        snap_subset,
        left_on=["player_display_name", "recent_team", "season", "week"],
        right_on=["snap_player_name", "team", "season", "week"],
        how="left",
        suffixes=("", "_snap"),
    )
    merged.drop(columns=["snap_player_name", "team_snap"], inplace=True, errors="ignore")

    return merged


def merge_ngs(merged: pd.DataFrame, ngs_type: str, ngs_df: pd.DataFrame,
              feature_cols: list[str]) -> pd.DataFrame:
    """
    Merge Next Gen Stats data for a specific stat type.

    Join logic: NGS uses player_display_name + team_abbr + season + week.
    """
    # Identify key columns present in the NGS data
    key_cols = ["player_display_name", "team_abbr", "season", "week"]
    available_keys = [c for c in key_cols if c in ngs_df.columns]
    available_features = [c for c in feature_cols if c in ngs_df.columns]

    if not available_features:
        print(f"  Warning: No NGS {ngs_type} feature columns found, skipping.")
        return merged

    ngs_subset = ngs_df[available_keys + available_features].copy()

    # Rename to avoid collisions, prefix with ngs type
    rename_map = {col: f"ngs_{ngs_type}_{col}" for col in available_features}
    ngs_subset.rename(columns=rename_map, inplace=True)

    # Build merge keys
    left_keys = ["player_display_name", "recent_team", "season", "week"]
    right_keys = ["player_display_name", "team_abbr", "season", "week"]

    # Only use keys that exist
    right_keys = [k for k in right_keys if k in ngs_subset.columns]

    merged = merged.merge(
        ngs_subset,
        left_on=left_keys[:len(right_keys)],
        right_on=right_keys,
        how="left",
        suffixes=("", f"_ngs_{ngs_type}"),
    )

    # Clean up duplicate key columns
    for col in [f"player_display_name_ngs_{ngs_type}", f"team_abbr"]:
        if col in merged.columns and col not in left_keys:
            merged.drop(columns=[col], inplace=True, errors="ignore")

    return merged


def merge_injuries(merged: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Merge injury report status onto the master DataFrame.

    Adds: report_status, practice_status (encoded numerically).

    Join logic: injuries use player name + team + season + week.
    """
    inj_cols = ["season", "week", "team", "full_name", "report_status",
                "practice_status"]
    # Fall back to available columns
    available = [c for c in inj_cols if c in injuries.columns]
    if "report_status" not in available:
        # Try alternate column names
        if "game_status" in injuries.columns:
            injuries["report_status"] = injuries["game_status"]
            available.append("report_status")

    inj_subset = injuries[available].copy()

    # Use the name column that exists
    name_col = "full_name" if "full_name" in inj_subset.columns else None
    if name_col is None:
        for candidate in ["player_name", "player", "name"]:
            if candidate in inj_subset.columns:
                name_col = candidate
                break

    if name_col is None:
        print("  Warning: No player name column found in injuries, skipping.")
        return merged

    merged = merged.merge(
        inj_subset,
        left_on=["player_display_name", "recent_team", "season", "week"],
        right_on=[name_col, "team", "season", "week"],
        how="left",
        suffixes=("", "_inj"),
    )

    # Clean up duplicate columns
    for col in [name_col, "team_inj", "team"]:
        if col in merged.columns and col not in ["season", "week"]:
            merged.drop(columns=[col], inplace=True, errors="ignore")

    # Encode injury status numerically
    status_map = {
        "Out": 0,
        "Doubtful": 1,
        "Questionable": 2,
        "Probable": 3,
    }
    if "report_status" in merged.columns:
        merged["injury_status_code"] = merged["report_status"].map(status_map)
        # Healthy players (no injury report) get highest value
        merged["injury_status_code"] = merged["injury_status_code"].fillna(4)

    practice_map = {
        "Did Not Participate": 0,
        "DNP": 0,
        "Limited Participation": 1,
        "Limited": 1,
        "Full Participation": 2,
        "Full": 2,
    }
    if "practice_status" in merged.columns:
        merged["practice_status_code"] = merged["practice_status"].map(practice_map)
        merged["practice_status_code"] = merged["practice_status_code"].fillna(2)

    return merged


def build_merged_dataset() -> pd.DataFrame:
    """
    Full merge pipeline: load all raw data and combine into one DataFrame.
    """
    from src.config import (
        NGS_PASSING_FEATURES, NGS_RUSHING_FEATURES, NGS_RECEIVING_FEATURES
    )

    print("Loading raw datasets...")
    weekly = load_raw("weekly.parquet")
    schedules = load_raw("schedules.parquet")
    snaps = load_raw("snap_counts.parquet")
    ngs_passing = load_raw("ngs_passing.parquet")
    ngs_rushing = load_raw("ngs_rushing.parquet")
    ngs_receiving = load_raw("ngs_receiving.parquet")
    injuries = load_raw("injuries.parquet")

    print(f"  weekly: {len(weekly)} rows")
    print(f"  schedules: {len(schedules)} rows")
    print(f"  snap_counts: {len(snaps)} rows")
    print(f"  ngs_passing: {len(ngs_passing)} rows")
    print(f"  ngs_rushing: {len(ngs_rushing)} rows")
    print(f"  ngs_receiving: {len(ngs_receiving)} rows")
    print(f"  injuries: {len(injuries)} rows")

    # Step 1: Start with weekly as base
    merged = weekly.copy()
    base_rows = len(merged)
    print(f"\nBase (weekly): {base_rows} rows")

    # Step 2: Merge schedules
    print("Merging schedules...")
    merged = merge_schedules(merged, schedules)
    print(f"  -> {len(merged)} rows (should be ~{base_rows})")

    # Step 3: Merge snap counts
    print("Merging snap counts...")
    merged = merge_snap_counts(merged, snaps)
    print(f"  -> {len(merged)} rows")

    # Step 4: Merge NGS data (one per stat type)
    print("Merging NGS passing...")
    merged = merge_ngs(merged, "passing", ngs_passing, NGS_PASSING_FEATURES)
    print(f"  -> {len(merged)} rows")

    print("Merging NGS rushing...")
    merged = merge_ngs(merged, "rushing", ngs_rushing, NGS_RUSHING_FEATURES)
    print(f"  -> {len(merged)} rows")

    print("Merging NGS receiving...")
    merged = merge_ngs(merged, "receiving", ngs_receiving, NGS_RECEIVING_FEATURES)
    print(f"  -> {len(merged)} rows")

    # Step 5: Merge injuries
    print("Merging injuries...")
    merged = merge_injuries(merged, injuries)
    print(f"  -> {len(merged)} rows")

    # Sanity check: row count should not have exploded
    if len(merged) > base_rows * 1.1:
        print(f"\n  WARNING: Row count increased by {len(merged) - base_rows} "
              f"({(len(merged)/base_rows - 1)*100:.1f}%). Check for duplicate merge keys.")

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "merged_player_weekly.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved merged dataset: {out_path}")
    print(f"Final shape: {merged.shape}")

    return merged
