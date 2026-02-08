"""
FastAPI backend — wraps the data pipeline steps as API endpoints.
Streams logs back to the frontend in real-time via SSE.
"""
import sys
import io
import time
import threading
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
