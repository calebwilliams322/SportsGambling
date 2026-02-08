"""
Data ingestion — pulls all datasets from nfl_data_py and caches as parquet.
"""
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from src.config import SEASONS, DATA_RAW


def _aggregate_pbp_to_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-by-play data into weekly player stats.
    Used for seasons where import_weekly_data is not yet available (e.g. 2025).
    Produces columns matching the import_weekly_data format.
    """
    rows = []

    # --- Passing stats ---
    passing = pbp[pbp["passer_player_id"].notna()].copy()
    pass_agg = passing.groupby(
        ["season", "week", "passer_player_id", "passer_player_name", "posteam"]
    ).agg(
        completions=("complete_pass", "sum"),
        attempts=("play_type", "count"),
        passing_yards=("passing_yards", "sum"),
        passing_tds=("pass_touchdown", "sum"),
        interceptions=("interception", "sum"),
        sacks=("sack", "sum"),
        passing_air_yards=("air_yards", "sum"),
        passing_yards_after_catch=("yards_after_catch", "sum"),
        passing_epa=("epa", "sum"),
        passing_first_downs=("first_down_pass", "sum"),
    ).reset_index()
    pass_agg.rename(columns={
        "passer_player_id": "player_id",
        "passer_player_name": "player_display_name",
        "posteam": "recent_team",
    }, inplace=True)
    pass_agg["position"] = "QB"
    rows.append(pass_agg)

    # --- Rushing stats ---
    rushing = pbp[(pbp["rusher_player_id"].notna()) & (pbp["play_type"] == "run")].copy()
    rush_agg = rushing.groupby(
        ["season", "week", "rusher_player_id", "rusher_player_name", "posteam"]
    ).agg(
        carries=("play_type", "count"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rush_touchdown", "sum"),
        rushing_epa=("epa", "sum"),
        rushing_first_downs=("first_down_rush", "sum"),
    ).reset_index()
    rush_agg.rename(columns={
        "rusher_player_id": "player_id",
        "rusher_player_name": "player_display_name",
        "posteam": "recent_team",
    }, inplace=True)
    rows.append(rush_agg)

    # --- Receiving stats ---
    receiving = pbp[
        (pbp["receiver_player_id"].notna()) & (pbp["play_type"] == "pass")
    ].copy()
    rec_agg = receiving.groupby(
        ["season", "week", "receiver_player_id", "receiver_player_name", "posteam"]
    ).agg(
        receptions=("complete_pass", "sum"),
        targets=("play_type", "count"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("touchdown", "sum"),
        receiving_air_yards=("air_yards", "sum"),
        receiving_yards_after_catch=("yards_after_catch", "sum"),
        receiving_epa=("epa", "sum"),
        receiving_first_downs=("first_down_pass", "sum"),
    ).reset_index()
    rec_agg.rename(columns={
        "receiver_player_id": "player_id",
        "receiver_player_name": "player_display_name",
        "posteam": "recent_team",
    }, inplace=True)
    rows.append(rec_agg)

    # Merge all stat types per player per week
    base = rows[0]  # passing
    base = base.merge(
        rows[1], on=["season", "week", "player_id", "player_display_name", "recent_team"],
        how="outer",
    )
    base = base.merge(
        rows[2], on=["season", "week", "player_id", "player_display_name", "recent_team"],
        how="outer",
    )

    # Fill NaN stats with 0
    stat_cols = [
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "sacks", "passing_air_yards", "passing_yards_after_catch", "passing_epa",
        "passing_first_downs",
        "carries", "rushing_yards", "rushing_tds", "rushing_epa", "rushing_first_downs",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
        "receiving_first_downs",
    ]
    for col in stat_cols:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    # Compute target_share and air_yards_share per team per week
    team_totals = base.groupby(["season", "week", "recent_team"]).agg(
        team_targets=("targets", "sum"),
        team_air_yards=("receiving_air_yards", "sum"),
    ).reset_index()
    base = base.merge(team_totals, on=["season", "week", "recent_team"], how="left")
    base["target_share"] = np.where(base["team_targets"] > 0,
                                     base["targets"] / base["team_targets"], 0)
    base["air_yards_share"] = np.where(base["team_air_yards"] > 0,
                                        base["receiving_air_yards"] / base["team_air_yards"], 0)
    base["wopr"] = 1.5 * base["target_share"] + 0.7 * base["air_yards_share"]
    base.drop(columns=["team_targets", "team_air_yards"], inplace=True)

    # Compute fantasy points (standard and PPR)
    base["fantasy_points"] = (
        base["passing_yards"] * 0.04
        + base["passing_tds"] * 4
        - base["interceptions"] * 1
        + base["rushing_yards"] * 0.1
        + base["rushing_tds"] * 6
        + base["receptions"] * 0  # standard = 0 per reception
        + base["receiving_yards"] * 0.1
        + base["receiving_tds"] * 6
    )
    base["fantasy_points_ppr"] = base["fantasy_points"] + base["receptions"] * 1

    return base


def pull_weekly_data(seasons=SEASONS) -> pd.DataFrame:
    """
    Player-level stats per week. One row per player per game.
    Falls back to aggregating play-by-play for seasons where
    import_weekly_data is not available.
    """
    # Split seasons into those with weekly data and those needing PBP fallback
    weekly_frames = []

    # Try weekly data first (faster, more complete)
    weekly_seasons = []
    pbp_seasons = []
    for s in seasons:
        try:
            nfl.import_weekly_data([s])
            weekly_seasons.append(s)
        except Exception:
            pbp_seasons.append(s)

    if weekly_seasons:
        print(f"Pulling weekly data for {weekly_seasons[0]}-{weekly_seasons[-1]}...")
        df = nfl.import_weekly_data(weekly_seasons)
        df = nfl.clean_nfl_data(df)
        weekly_frames.append(df)

    if pbp_seasons:
        print(f"Weekly data unavailable for {pbp_seasons} — aggregating from play-by-play...")
        pbp = nfl.import_pbp_data(pbp_seasons)
        pbp = nfl.clean_nfl_data(pbp)
        agg = _aggregate_pbp_to_weekly(pbp)
        agg = nfl.clean_nfl_data(agg)
        weekly_frames.append(agg)

    return pd.concat(weekly_frames, ignore_index=True)


def pull_schedules(seasons=SEASONS) -> pd.DataFrame:
    """Game-level schedule data with spreads, totals, weather."""
    print(f"Pulling schedules for {seasons[0]}-{seasons[-1]}...")
    df = nfl.import_schedules(seasons)
    df = nfl.clean_nfl_data(df)
    return df


def pull_snap_counts(seasons=SEASONS) -> pd.DataFrame:
    """Weekly snap count percentages per player."""
    print(f"Pulling snap counts for {seasons[0]}-{seasons[-1]}...")
    df = nfl.import_snap_counts(seasons)
    df = nfl.clean_nfl_data(df)
    return df


def pull_ngs_data(seasons=SEASONS) -> dict[str, pd.DataFrame]:
    """Next Gen Stats tracking data for passing, rushing, receiving."""
    ngs = {}
    for stat_type in ["passing", "rushing", "receiving"]:
        frames = []
        for s in seasons:
            try:
                df = nfl.import_ngs_data(stat_type, [s])
                frames.append(df)
            except Exception:
                pass
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = nfl.clean_nfl_data(combined)
            avail = sorted(combined["season"].unique())
            print(f"Pulled NGS {stat_type} data for seasons {avail[0]}-{avail[-1]}")
        else:
            combined = pd.DataFrame()
            print(f"No NGS {stat_type} data available")
        ngs[stat_type] = combined
    return ngs


def pull_injuries(seasons=SEASONS) -> pd.DataFrame:
    """Weekly injury reports. Skips seasons that aren't available."""
    frames = []
    for s in seasons:
        try:
            df = nfl.import_injuries([s])
            frames.append(df)
        except Exception:
            pass
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = nfl.clean_nfl_data(combined)
        avail = sorted(combined["season"].unique())
        print(f"Pulled injury data for seasons {avail[0]}-{avail[-1]}")
    else:
        combined = pd.DataFrame()
        print("No injury data available")
    return combined


def pull_ids() -> pd.DataFrame:
    """Cross-reference ID mapping table for merging across datasets."""
    print("Pulling player ID crosswalk...")
    return nfl.import_ids()


def download_all(seasons=SEASONS):
    """Pull all datasets and save as parquet files in data/raw/."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    weekly = pull_weekly_data(seasons)
    weekly.to_parquet(DATA_RAW / "weekly.parquet", index=False)
    print(f"  -> weekly.parquet: {len(weekly)} rows, {len(weekly.columns)} cols")

    schedules = pull_schedules(seasons)
    schedules.to_parquet(DATA_RAW / "schedules.parquet", index=False)
    print(f"  -> schedules.parquet: {len(schedules)} rows")

    snaps = pull_snap_counts(seasons)
    snaps.to_parquet(DATA_RAW / "snap_counts.parquet", index=False)
    print(f"  -> snap_counts.parquet: {len(snaps)} rows")

    ngs = pull_ngs_data(seasons)
    for stat_type, df in ngs.items():
        df.to_parquet(DATA_RAW / f"ngs_{stat_type}.parquet", index=False)
        print(f"  -> ngs_{stat_type}.parquet: {len(df)} rows")

    injuries = pull_injuries(seasons)
    injuries.to_parquet(DATA_RAW / "injuries.parquet", index=False)
    print(f"  -> injuries.parquet: {len(injuries)} rows")

    ids_df = pull_ids()
    ids_df.to_parquet(DATA_RAW / "player_ids.parquet", index=False)
    print(f"  -> player_ids.parquet: {len(ids_df)} rows")

    print("\nAll data downloaded successfully.")
