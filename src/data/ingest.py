"""
Data ingestion — pulls all datasets from nfl_data_py and caches as parquet.
"""
import nfl_data_py as nfl
import pandas as pd
from src.config import SEASONS, DATA_RAW


def pull_weekly_data(seasons=SEASONS) -> pd.DataFrame:
    """Player-level stats per week. One row per player per game."""
    print(f"Pulling weekly data for {seasons[0]}-{seasons[-1]}...")
    df = nfl.import_weekly_data(seasons)
    df = nfl.clean_nfl_data(df)
    return df


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
        print(f"Pulling NGS {stat_type} data for {seasons[0]}-{seasons[-1]}...")
        df = nfl.import_ngs_data(stat_type, seasons)
        df = nfl.clean_nfl_data(df)
        ngs[stat_type] = df
    return ngs


def pull_injuries(seasons=SEASONS) -> pd.DataFrame:
    """Weekly injury reports."""
    print(f"Pulling injury data for {seasons[0]}-{seasons[-1]}...")
    df = nfl.import_injuries(seasons)
    df = nfl.clean_nfl_data(df)
    return df


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
