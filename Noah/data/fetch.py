import os
import pandas as pd
import nfl_data_py as nfl

from config import DATA_DIR

# nfl_data_py only has through 2024; nflreadpy has 2025
NFL_DATA_PY_YEARS = list(range(2015, 2025))
NFLREADPY_YEARS = [2025]


def _cache_path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _fetch_2025_weekly():
    """Fetch 2025 weekly data via nflreadpy (the successor package)."""
    import nflreadpy as nflr
    df = nflr.load_player_stats(seasons=NFLREADPY_YEARS).to_pandas()
    return df


def fetch_weekly_stats(force=False):
    """Weekly player-level stats from 2015-2025."""
    path = _cache_path("weekly_stats_with_2025")
    if not force and os.path.exists(path):
        return pd.read_parquet(path)

    # 2015-2024 from nfl_data_py
    df_old = nfl.import_weekly_data(NFL_DATA_PY_YEARS, downcast=True)

    # 2025 from nflreadpy — rename columns to match nfl_data_py
    df_new = _fetch_2025_weekly()
    df_new = df_new.rename(columns={
        "passing_interceptions": "interceptions",
        "team": "recent_team",
    })

    # Align columns — keep only columns present in both
    common_cols = list(set(df_old.columns) & set(df_new.columns))
    df = pd.concat([df_old[common_cols], df_new[common_cols]], ignore_index=True)

    df.to_parquet(path, index=False)
    return df


def fetch_schedules(force=False):
    """Game-level schedule data with home/away and scores."""
    path = _cache_path("schedules_with_2025")
    if not force and os.path.exists(path):
        return pd.read_parquet(path)

    # nfl_data_py schedules cover through current season
    all_years = NFL_DATA_PY_YEARS + NFLREADPY_YEARS
    df = nfl.import_schedules(all_years)
    df.to_parquet(path, index=False)
    return df
