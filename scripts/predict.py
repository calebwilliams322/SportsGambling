#!/usr/bin/env python3
"""
Generate predictions for player props using a trained model.

Usage:
    python scripts/predict.py --stat passing_yards
    python scripts/predict.py --stat passing_yards --players "Patrick Mahomes,Jalen Hurts"
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, MODELS_DIR, TARGET_STATS
from src.models.player_props import PlayerPropsNet
from sklearn.preprocessing import StandardScaler


def load_model_and_meta(stat_name: str):
    """Load a trained model and its metadata."""
    meta_path = MODELS_DIR / f"{stat_name}_meta.json"
    model_path = MODELS_DIR / f"{stat_name}_best.pt"

    if not meta_path.exists() or not model_path.exists():
        print(f"No trained model found for {stat_name}.")
        print(f"Run: python scripts/train.py --stat {stat_name}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    model = PlayerPropsNet(input_size=meta["n_features"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model, meta


def predict_players(stat_name: str, player_names: list[str] = None):
    """
    Generate predictions for players using the most recent available data.
    """
    model, meta = load_model_and_meta(stat_name)
    feature_cols = meta["feature_columns"]

    # Load the full feature dataset
    df = pd.read_parquet(DATA_PROCESSED / "features_player_weekly.parquet")

    # Get the most recent data for each player
    df = df.sort_values(["player_id", "season", "week"])
    latest = df.groupby("player_id").last().reset_index()

    if player_names:
        # Filter to requested players
        latest = latest[latest["player_display_name"].isin(player_names)]
        if latest.empty:
            print(f"No matching players found. Available players in dataset:")
            all_names = df["player_display_name"].unique()
            for name in sorted(all_names)[:20]:
                print(f"  {name}")
            print("  ...")
            return

    # Filter to players who have data for this stat
    latest = latest[latest[stat_name].notna()]

    # Extract features
    available_cols = [c for c in feature_cols if c in latest.columns]
    missing_cols = [c for c in feature_cols if c not in latest.columns]
    if missing_cols:
        print(f"Warning: {len(missing_cols)} feature columns missing, filling with 0")
        for col in missing_cols:
            latest[col] = 0.0

    X = latest[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    # Normalize (using training data stats — ideally we'd save the scaler,
    # but for now we standardize based on the feature values)
    scaler = StandardScaler()
    # Load train data to fit scaler properly
    full_df = pd.read_parquet(DATA_PROCESSED / "features_player_weekly.parquet")
    from src.config import TRAIN_SEASONS
    train_df = full_df[full_df["season"].isin(TRAIN_SEASONS)]
    train_X = train_df[feature_cols].values.astype(np.float32)
    train_X = np.nan_to_num(train_X, nan=0.0)
    scaler.fit(train_X)

    X_scaled = scaler.transform(X)

    # Predict
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    # Build results
    results = pd.DataFrame({
        "player": latest["player_display_name"].values,
        "team": latest["recent_team"].values,
        "position": latest["position"].values if "position" in latest.columns else "N/A",
        f"predicted_{stat_name}": np.round(preds, 1),
        f"last_game_{stat_name}": latest[stat_name].values,
    })

    results = results.sort_values(f"predicted_{stat_name}", ascending=False)

    print(f"\n{'=' * 60}")
    print(f"Predictions: {stat_name}")
    print(f"{'=' * 60}")
    print(results.to_string(index=False))
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate player prop predictions")
    parser.add_argument("--stat", type=str, required=True, help="Stat to predict")
    parser.add_argument("--players", type=str, default=None,
                        help="Comma-separated player names")
    parser.add_argument("--top", type=int, default=30,
                        help="Show top N predictions (default: 30)")
    args = parser.parse_args()

    if args.stat not in TARGET_STATS:
        print(f"Unknown stat: {args.stat}")
        print(f"Available: {TARGET_STATS}")
        sys.exit(1)

    player_names = None
    if args.players:
        player_names = [p.strip() for p in args.players.split(",")]

    results = predict_players(args.stat, player_names)
    if results is not None and args.top and not args.players:
        print(f"(Showing top {args.top} — use --players to filter)")


if __name__ == "__main__":
    main()
