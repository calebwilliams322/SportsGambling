"""
FastAPI backend — wraps the data pipeline steps as API endpoints.
Streams logs back to the frontend in real-time via SSE.
"""
import sys
import os
import io
import json
import time
import threading
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
import urllib.request
import urllib.parse

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = FastAPI(title="SportsBetting Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track pipeline state
pipeline_state = {
    "download": {"status": "idle", "logs": ""},
    "build_features": {"status": "idle", "logs": ""},
    "train": {"status": "idle", "logs": ""},
}


class LogCapture(io.StringIO):
    """Captures print output and stores it in pipeline_state."""

    def __init__(self, step_name: str):
        super().__init__()
        self.step_name = step_name

    def write(self, s):
        if s.strip():
            pipeline_state[self.step_name]["logs"] += s
        return super().write(s)


def run_step_in_thread(step_name: str, func):
    """Run a pipeline step in a background thread, capturing output."""
    pipeline_state[step_name]["status"] = "running"
    pipeline_state[step_name]["logs"] = ""

    def _run():
        capture = LogCapture(step_name)
        try:
            with redirect_stdout(capture), redirect_stderr(capture):
                func()
            pipeline_state[step_name]["status"] = "done"
        except Exception as e:
            pipeline_state[step_name]["logs"] += f"\nERROR: {str(e)}"
            pipeline_state[step_name]["status"] = "error"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


@app.get("/api/status")
def get_status():
    """Get the current status of all pipeline steps."""
    from src.config import DATA_RAW, DATA_PROCESSED

    # Check what data files exist
    raw_files = list(DATA_RAW.glob("*.parquet")) if DATA_RAW.exists() else []
    processed_files = list(DATA_PROCESSED.glob("*.parquet")) if DATA_PROCESSED.exists() else []

    return {
        "steps": {
            "download": {
                "status": pipeline_state["download"]["status"],
                "files": [f.name for f in raw_files],
                "has_data": len(raw_files) > 0,
            },
            "build_features": {
                "status": pipeline_state["build_features"]["status"],
                "files": [f.name for f in processed_files],
                "has_data": len(processed_files) > 0,
            },
        },
    }


@app.post("/api/download")
def start_download():
    """Step 1: Download all raw data from nfl_data_py."""
    if pipeline_state["download"]["status"] == "running":
        return {"message": "Already running"}

    from src.data.ingest import download_all
    from src.config import SEASONS

    run_step_in_thread("download", lambda: download_all(SEASONS))
    return {"message": "Download started"}


@app.post("/api/build-features")
def start_build_features():
    """Step 2: Merge datasets and engineer features."""
    if pipeline_state["build_features"]["status"] == "running":
        return {"message": "Already running"}

    def _build():
        from src.data.merge import build_merged_dataset
        from src.data.features import build_features
        build_merged_dataset()
        build_features()

    run_step_in_thread("build_features", _build)
    return {"message": "Feature building started"}


@app.get("/api/logs/{step_name}")
def get_logs(step_name: str):
    """Stream logs for a pipeline step."""
    if step_name not in pipeline_state:
        return {"error": "Unknown step"}

    def generate():
        last_len = 0
        while True:
            logs = pipeline_state[step_name]["logs"]
            if len(logs) > last_len:
                new_content = logs[last_len:]
                last_len = len(logs)
                yield f"data: {new_content}\n\n"
            status = pipeline_state[step_name]["status"]
            if status in ("done", "error", "idle"):
                yield f"event: status\ndata: {status}\n\n"
                break
            time.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/data-preview/{dataset}")
def preview_data(dataset: str, rows: int = 20):
    """Preview a dataset (first N rows)."""
    from src.config import DATA_RAW, DATA_PROCESSED
    import pandas as pd

    file_map = {
        "weekly": DATA_RAW / "weekly.parquet",
        "schedules": DATA_RAW / "schedules.parquet",
        "snap_counts": DATA_RAW / "snap_counts.parquet",
        "ngs_passing": DATA_RAW / "ngs_passing.parquet",
        "ngs_rushing": DATA_RAW / "ngs_rushing.parquet",
        "ngs_receiving": DATA_RAW / "ngs_receiving.parquet",
        "injuries": DATA_RAW / "injuries.parquet",
        "merged": DATA_PROCESSED / "merged_player_weekly.parquet",
        "features": DATA_PROCESSED / "features_player_weekly.parquet",
    }

    path = file_map.get(dataset)
    if not path or not path.exists():
        return {"error": f"Dataset '{dataset}' not found"}

    df = pd.read_parquet(path)

    return {
        "name": dataset,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "preview": df.head(rows).fillna("").to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


@app.get("/api/data-stats")
def data_stats():
    """Get summary statistics for the processed feature dataset."""
    from src.config import DATA_PROCESSED
    import pandas as pd

    path = DATA_PROCESSED / "features_player_weekly.parquet"
    if not path.exists():
        return {"error": "Features not built yet"}

    df = pd.read_parquet(path)

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "seasons": sorted([int(s) for s in df["season"].unique()]),
        "latest_week": int(df[df["season"] == df["season"].max()]["week"].max()),
        "players": int(df["player_display_name"].nunique()),
        "top_passers_2025": (
            df[df["season"] == 2025]
            .groupby("player_display_name")["passing_yards"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        ) if 2025 in df["season"].values else {},
    }


# ---- Training & Prediction request models ----

class TrainRequest(BaseModel):
    stat: str
    learning_rate: Optional[float] = None
    hidden_sizes: Optional[list[int]] = None
    dropout: Optional[float] = None
    patience: Optional[int] = None


class PredictRequest(BaseModel):
    stat: str
    player_name: str
    opponent: str
    is_home: int
    spread: float
    total_line: float
    is_dome: int
    temp: float
    wind: float


class BatchPredictRequest(BaseModel):
    stat: str
    home_team: str
    away_team: str
    spread: float
    total_line: float
    is_dome: int
    temp: float
    wind: float


# ---- New endpoints ----

@app.get("/api/models")
def get_models():
    """Return list of trained models by scanning models/*_meta.json."""
    from src.config import MODELS_DIR

    models = []
    if not MODELS_DIR.exists():
        return {"models": []}

    for meta_path in sorted(MODELS_DIR.glob("*_meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        models.append({
            "stat": meta["stat"],
            "n_features": meta["n_features"],
            "test_mae": meta.get("test_metrics", {}).get("mae"),
            "improvement_pct": meta.get("baseline_comparison", {}).get("improvement_pct"),
            "epochs_trained": meta.get("history", {}).get("epochs_trained"),
            "config": meta.get("best_config", {}),
        })

    return {"models": models}


@app.post("/api/train")
def start_train(req: TrainRequest):
    """Train (or retrain) a model for a stat type."""
    from src.config import TARGET_STATS, DATA_PROCESSED

    if not (DATA_PROCESSED / "features_player_weekly.parquet").exists():
        return {"error": "Features not built yet. Run the data pipeline first."}

    if req.stat not in TARGET_STATS:
        return {"error": f"Unknown stat. Available: {', '.join(TARGET_STATS)}"}

    if pipeline_state["train"]["status"] == "running":
        return {"error": "Training already in progress. Wait for it to finish."}

    def _train():
        from src.models.player_props import PlayerPropsNet
        from src.training.dataset import prepare_data
        from src.training.trainer import Trainer
        from src.training.evaluate import evaluate_model, compare_to_baseline
        from src.config import MODELS_DIR

        print(f"Training model for: {req.stat}")
        print(f"{'=' * 50}")

        # Prepare data
        train_loader, val_loader, test_loader, scaler, feature_cols, target_info = prepare_data(req.stat)

        # Build model with optional custom hyperparams
        input_size = len(feature_cols)
        model = PlayerPropsNet(
            input_size=input_size,
            hidden_sizes=req.hidden_sizes,
            dropout=req.dropout,
        )
        print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

        # Train
        trainer = Trainer(
            model,
            learning_rate=req.learning_rate,
            patience=req.patience,
        )
        history = trainer.fit(train_loader, val_loader, stat_name=req.stat)

        # Evaluate
        results = evaluate_model(model, test_loader, device=trainer.device,
                                 stat_name=req.stat, target_stats=target_info)
        baseline = compare_to_baseline(results["predictions"], results["actuals"], req.stat)

        # Save metadata
        meta = {
            "stat": req.stat,
            "n_features": input_size,
            "feature_columns": feature_cols,
            "target_normalization": target_info,
            "best_config": {
                "learning_rate": req.learning_rate or 1e-3,
                "hidden_sizes": req.hidden_sizes or [128, 64, 32],
                "dropout": req.dropout or 0.3,
                "patience": req.patience or 10,
            },
            "history": {
                "best_val_loss": history["best_val_loss"],
                "epochs_trained": history["epochs_trained"],
            },
            "test_metrics": results["metrics"],
            "baseline_comparison": baseline,
        }

        meta_path = MODELS_DIR / f"{req.stat}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\nSaved metadata to {meta_path}")
        print("Training complete!")

    run_step_in_thread("train", _train)
    return {"message": f"Training started for {req.stat}"}


@app.get("/api/players")
def get_players(stat: str, team: str = None):
    """Return eligible players for a stat type, optionally filtered by team(s)."""
    from src.config import DATA_PROCESSED, TARGET_STATS, STAT_POSITION_FILTER
    import pandas as pd

    if stat not in TARGET_STATS:
        return {"error": f"Unknown stat. Available: {', '.join(TARGET_STATS)}"}

    features_path = DATA_PROCESSED / "features_player_weekly.parquet"
    if not features_path.exists():
        return {"error": "Features not built yet. Run the data pipeline first."}

    df = pd.read_parquet(features_path)

    # Filter to valid positions for this stat
    positions = STAT_POSITION_FILTER.get(stat, [])
    if positions and "position" in df.columns:
        df = df[df["position"].isin(positions)]

    # Filter by team(s) if provided
    if team:
        teams = [t.strip() for t in team.split(",")]
        df = df[df["recent_team"].isin(teams)]

    # Get latest season
    latest_season = df["season"].max()
    df = df[df["season"] == latest_season]
    max_week = int(df["week"].max())

    # Recency filter: only players active within last 5 weeks
    recency_cutoff = max_week - 5
    recent_per_player = df.groupby("player_id")["week"].max().reset_index()
    recent_ids = recent_per_player[recent_per_player["week"] >= recency_cutoff]["player_id"]
    df = df[df["player_id"].isin(recent_ids)]

    # Get unique players (most recent row per player)
    df = df.sort_values(["player_id", "week"]).groupby("player_id").last().reset_index()

    players = []
    for _, row in df.iterrows():
        players.append({
            "name": row.get("player_display_name", "Unknown"),
            "team": row.get("recent_team", ""),
            "position": row.get("position", ""),
        })

    players.sort(key=lambda p: p["name"])
    return {"players": players, "positions": positions}


@app.get("/api/schedule-games")
def get_schedule_games():
    """Return playoff/Super Bowl games from schedules for game context auto-fill."""
    from src.config import DATA_RAW
    import pandas as pd

    sched_path = DATA_RAW / "schedules.parquet"
    if not sched_path.exists():
        return {"error": "Schedules not downloaded yet. Run the data pipeline first."}

    sched = pd.read_parquet(sched_path)

    # Get the latest season
    latest_season = sched["season"].max()
    sched = sched[sched["season"] == latest_season]

    # Filter to playoff/Super Bowl games (game_type != 'REG')
    if "game_type" in sched.columns:
        playoff = sched[sched["game_type"] != "REG"]
        if len(playoff) > 0:
            sched = playoff

    # Sort by week descending so most recent (e.g. Super Bowl) is first
    sched = sched.sort_values("week", ascending=False)

    games = []
    for _, row in sched.iterrows():
        games.append({
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "week": int(row.get("week", 0)),
            "game_type": row.get("game_type", ""),
            "spread_line": float(row["spread_line"]) if pd.notna(row.get("spread_line")) else 0.0,
            "total_line": float(row["total_line"]) if pd.notna(row.get("total_line")) else 0.0,
            "roof": row.get("roof", ""),
            "temp": float(row["temp"]) if pd.notna(row.get("temp")) else 72.0,
            "wind": float(row["wind"]) if pd.notna(row.get("wind")) else 0.0,
        })

    return {"games": games}


@app.post("/api/predict")
def run_prediction(req: PredictRequest):
    """Run a single-player prediction with game context overrides."""
    from src.config import MODELS_DIR, DATA_PROCESSED, TARGET_STATS, STAT_POSITION_FILTER
    import pandas as pd
    import numpy as np
    import torch
    from src.models.player_props import PlayerPropsNet
    from sklearn.preprocessing import StandardScaler

    if req.stat not in TARGET_STATS:
        return {"error": f"Unknown stat. Available: {', '.join(TARGET_STATS)}"}

    meta_path = MODELS_DIR / f"{req.stat}_meta.json"
    model_path = MODELS_DIR / f"{req.stat}_best.pt"

    if not meta_path.exists() or not model_path.exists():
        return {"error": f"No trained model for {req.stat}. Train it first."}

    features_path = DATA_PROCESSED / "features_player_weekly.parquet"
    if not features_path.exists():
        return {"error": "Features not built yet. Run the data pipeline first."}

    # Load model + meta
    with open(meta_path) as f:
        meta = json.load(f)

    model = PlayerPropsNet(input_size=meta["n_features"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    feature_cols = meta["feature_columns"]
    target_norm = meta["target_normalization"]

    # Load features and find the player
    df = pd.read_parquet(features_path)
    df = df.sort_values(["player_id", "season", "week"])

    # Find the player's most recent row
    player_rows = df[df["player_display_name"] == req.player_name]
    if player_rows.empty:
        # Fuzzy suggest
        from difflib import get_close_matches
        all_names = df["player_display_name"].unique().tolist()
        suggestions = get_close_matches(req.player_name, all_names, n=5, cutoff=0.4)
        suggestion_str = ", ".join(suggestions) if suggestions else "no close matches"
        return {"error": f"Player '{req.player_name}' not found. Did you mean: {suggestion_str}"}

    player_row = player_rows.iloc[-1].copy()

    # Check position eligibility
    positions = STAT_POSITION_FILTER.get(req.stat, [])
    player_pos = player_row.get("position", "")
    if positions and player_pos not in positions:
        return {"error": f"Player {req.player_name} ({player_pos}) not eligible for {req.stat} predictions (requires {', '.join(positions)})"}

    # Override game context fields
    player_row["is_home"] = req.is_home
    player_row["team_spread"] = req.spread
    player_row["total_line"] = req.total_line
    player_row["is_dome"] = req.is_dome
    player_row["temp"] = req.temp
    player_row["wind"] = req.wind

    # Build feature row
    for col in feature_cols:
        if col not in player_row.index:
            player_row[col] = 0.0

    X = player_row[feature_cols].values.astype(np.float64).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    # Fit scaler on training data
    full_df = pd.read_parquet(features_path)

    # Filter same positions for consistent scaling
    if positions and "position" in full_df.columns:
        full_df = full_df[full_df["position"].isin(positions)]

    train_df = full_df  # fit scaler on all data (random split uses all seasons)
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0.0
    train_X = train_df[feature_cols].values.astype(np.float32)
    train_X = np.nan_to_num(train_X, nan=0.0)

    scaler = StandardScaler()
    scaler.fit(train_X)
    X_scaled = scaler.transform(X)

    # Run inference
    with torch.no_grad():
        pred_norm = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()[0]

    # Denormalize
    predicted_value = float(pred_norm * target_norm["std"] + target_norm["mean"])

    # Get recent rolling average for comparison
    rolling_col = f"{req.stat}_avg_3"
    rolling_avg = float(player_row.get(rolling_col, 0)) if rolling_col in player_row.index and pd.notna(player_row.get(rolling_col)) else None

    return {
        "predicted_value": round(predicted_value, 1),
        "player_name": req.player_name,
        "team": str(player_row.get("recent_team", "")),
        "position": str(player_row.get("position", "")),
        "opponent": req.opponent,
        "stat": req.stat,
        "rolling_avg": round(rolling_avg, 1) if rolling_avg is not None else None,
    }


@app.post("/api/predict-batch")
def run_batch_prediction(req: BatchPredictRequest):
    """Run predictions for all starters on both teams for a given stat + game context."""
    from src.config import MODELS_DIR, DATA_PROCESSED, TARGET_STATS, STAT_POSITION_FILTER
    import pandas as pd
    import numpy as np
    import torch
    from src.models.player_props import PlayerPropsNet
    from sklearn.preprocessing import StandardScaler

    if req.stat not in TARGET_STATS:
        return {"error": f"Unknown stat. Available: {', '.join(TARGET_STATS)}"}

    meta_path = MODELS_DIR / f"{req.stat}_meta.json"
    model_path = MODELS_DIR / f"{req.stat}_best.pt"

    if not meta_path.exists() or not model_path.exists():
        return {"error": f"No trained model for {req.stat}. Train it first."}

    features_path = DATA_PROCESSED / "features_player_weekly.parquet"
    if not features_path.exists():
        return {"error": "Features not built yet."}

    with open(meta_path) as f:
        meta = json.load(f)

    model = PlayerPropsNet(input_size=meta["n_features"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    feature_cols = meta["feature_columns"]
    target_norm = meta["target_normalization"]

    df = pd.read_parquet(features_path)
    df = df.sort_values(["player_id", "season", "week"])

    positions = STAT_POSITION_FILTER.get(req.stat, [])

    # Filter to both teams, eligible positions, latest season
    teams_df = df[df["recent_team"].isin([req.home_team, req.away_team])]
    if positions and "position" in teams_df.columns:
        teams_df = teams_df[teams_df["position"].isin(positions)]

    latest_season = teams_df["season"].max()
    teams_df = teams_df[teams_df["season"] == latest_season]
    max_week = int(teams_df["week"].max())

    # Recency filter: only players who played for this team within the last 8 weeks.
    # This filters out traded, cut, and practice-squad players automatically.
    recency_cutoff = max_week - 5
    recent_per_player = teams_df.groupby("player_id")["week"].max().reset_index()
    recent_ids = recent_per_player[recent_per_player["week"] >= recency_cutoff]["player_id"]
    teams_df = teams_df[teams_df["player_id"].isin(recent_ids)]

    # Get most recent row per player
    players_df = teams_df.sort_values(["player_id", "week"]).groupby("player_id").last().reset_index()

    # Filter to starters: players with a non-zero 3-game rolling avg
    rolling_col = f"{req.stat}_avg_3"
    if rolling_col in players_df.columns:
        players_df = players_df[players_df[rolling_col].notna() & (players_df[rolling_col] > 0)]

    if players_df.empty:
        return {"error": f"No eligible players found for {req.stat} on {req.home_team}/{req.away_team}"}

    # Fit scaler on all data (same as single predict)
    full_df = pd.read_parquet(features_path)
    if positions and "position" in full_df.columns:
        full_df = full_df[full_df["position"].isin(positions)]
    for col in feature_cols:
        if col not in full_df.columns:
            full_df[col] = 0.0
    scaler = StandardScaler()
    scaler.fit(np.nan_to_num(full_df[feature_cols].values.astype(np.float32), nan=0.0))

    # Run predictions for all players
    results = []
    for _, row in players_df.iterrows():
        player_row = row.copy()
        team = str(player_row.get("recent_team", ""))
        is_home = 1 if team == req.home_team else 0
        opponent = req.away_team if is_home else req.home_team
        spread = req.spread if is_home else -req.spread

        player_row["is_home"] = is_home
        player_row["team_spread"] = spread
        player_row["total_line"] = req.total_line
        player_row["is_dome"] = req.is_dome
        player_row["temp"] = req.temp
        player_row["wind"] = req.wind

        for col in feature_cols:
            if col not in player_row.index:
                player_row[col] = 0.0

        X = player_row[feature_cols].values.astype(np.float64).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        X_scaled = scaler.transform(X)

        with torch.no_grad():
            pred_norm = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()[0]

        predicted_value = float(pred_norm * target_norm["std"] + target_norm["mean"])
        rolling_avg = float(player_row.get(rolling_col, 0)) if rolling_col in player_row.index and pd.notna(player_row.get(rolling_col)) else None
        rolling_std_col = f"{req.stat}_std_3"
        rolling_std = float(player_row.get(rolling_std_col, 0)) if rolling_std_col in player_row.index and pd.notna(player_row.get(rolling_std_col)) else None

        results.append({
            "player_name": str(player_row.get("player_display_name", "Unknown")),
            "team": team,
            "position": str(player_row.get("position", "")),
            "opponent": opponent,
            "is_home": is_home,
            "predicted_value": round(predicted_value, 1),
            "rolling_avg": round(rolling_avg, 1) if rolling_avg is not None else None,
            "rolling_std": round(rolling_std, 1) if rolling_std is not None else None,
        })

    # Sort by predicted value descending
    results.sort(key=lambda r: r["predicted_value"], reverse=True)

    return {"stat": req.stat, "predictions": results}


# ---- Sportsbook Props via The Odds API ----

STAT_TO_MARKET = {
    "passing_yards": "player_pass_yds",
    "passing_tds": "player_pass_tds",
    "rushing_yards": "player_rush_yds",
    "carries": "player_rush_attempts",
    "receptions": "player_receptions",
    "receiving_yards": "player_reception_yds",
    "receiving_tds": "player_reception_tds",
}

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _odds_api_get(path: str, params: dict) -> dict | list | None:
    """Make a GET request to The Odds API."""
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return None
    params["apiKey"] = api_key
    qs = urllib.parse.urlencode(params)
    url = f"{ODDS_API_BASE}{path}?{qs}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Odds API error: {e}")
        return None


@app.get("/api/props")
def get_props(stat: str, home_team: str, away_team: str):
    """Fetch sportsbook player prop lines from The Odds API for a game."""
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return {"error": "ODDS_API_KEY not set. Export it as an environment variable.", "props": []}

    market = STAT_TO_MARKET.get(stat)
    if not market:
        return {"error": f"No market mapping for stat '{stat}'", "props": []}

    # Step 1: Get event IDs for NFL
    events = _odds_api_get("/sports/americanfootball_nfl/events", {"regions": "us"})
    if not events:
        return {"error": "Could not fetch NFL events from Odds API", "props": []}

    # Find the matching game by team names
    # The Odds API uses full team names (e.g., "Kansas City Chiefs")
    event_id = None
    for event in events:
        h = event.get("home_team", "").lower()
        a = event.get("away_team", "").lower()
        if (home_team.lower() in h or away_team.lower() in a or
            home_team.lower() in a or away_team.lower() in h):
            event_id = event["id"]
            break

    # Also try matching by abbreviation in team name
    if not event_id:
        # Map common abbreviations to city/team names
        abbrev_map = {
            "KC": "kansas city", "PHI": "philadelphia", "BUF": "buffalo",
            "SF": "san francisco", "DAL": "dallas", "GB": "green bay",
            "DET": "detroit", "BAL": "baltimore", "HOU": "houston",
            "MIN": "minnesota", "LA": "los angeles rams", "LAC": "los angeles chargers",
            "DEN": "denver", "MIA": "miami", "TB": "tampa bay",
            "CIN": "cincinnati", "PIT": "pittsburgh", "SEA": "seattle",
            "WAS": "washington", "ATL": "atlanta", "NO": "new orleans",
            "NYG": "new york giants", "NYJ": "new york jets",
            "NE": "new england", "ARI": "arizona", "CHI": "chicago",
            "CLE": "cleveland", "IND": "indianapolis", "CAR": "carolina",
            "JAX": "jacksonville", "TEN": "tennessee", "LV": "las vegas",
        }
        home_full = abbrev_map.get(home_team, home_team).lower()
        away_full = abbrev_map.get(away_team, away_team).lower()

        for event in events:
            h = event.get("home_team", "").lower()
            a = event.get("away_team", "").lower()
            if (home_full in h or home_full in a) and (away_full in h or away_full in a):
                event_id = event["id"]
                break

    if not event_id:
        return {"error": f"Game {away_team} @ {home_team} not found in Odds API events", "props": []}

    # Step 2: Get player props for this event
    odds_data = _odds_api_get(
        f"/sports/americanfootball_nfl/events/{event_id}/odds",
        {"regions": "us", "markets": market, "oddsFormat": "american"}
    )

    if not odds_data:
        return {"error": "Could not fetch props from Odds API", "props": []}

    # Step 3: Parse the response — extract O/U lines per player
    props = []
    bookmakers = odds_data.get("bookmakers", [])

    # Use first available bookmaker (usually DraftKings or FanDuel)
    bookmaker_name = ""
    for bk in bookmakers:
        bookmaker_name = bk.get("title", bk.get("key", ""))
        for mkt in bk.get("markets", []):
            if mkt.get("key") != market:
                continue
            for outcome in mkt.get("outcomes", []):
                if outcome.get("name") == "Over":
                    player_name = outcome.get("description", "")
                    line = outcome.get("point")
                    odds = outcome.get("price", 0)
                    if player_name and line is not None:
                        props.append({
                            "player_name": player_name,
                            "line": float(line),
                            "over_odds": odds,
                            "bookmaker": bookmaker_name,
                        })
        if props:
            break  # use first bookmaker that has data

    # Add under odds
    for bk in bookmakers:
        if bk.get("title", bk.get("key", "")) != bookmaker_name:
            continue
        for mkt in bk.get("markets", []):
            if mkt.get("key") != market:
                continue
            for outcome in mkt.get("outcomes", []):
                if outcome.get("name") == "Under":
                    player_name = outcome.get("description", "")
                    odds = outcome.get("price", 0)
                    for prop in props:
                        if prop["player_name"] == player_name:
                            prop["under_odds"] = odds
                            break

    return {"props": props, "bookmaker": bookmaker_name, "market": market}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
