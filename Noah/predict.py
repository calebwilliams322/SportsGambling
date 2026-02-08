import os
import sys
import torch
import pickle
import pandas as pd
import numpy as np

from config import CHECKPOINT_DIR, PROP_TYPES, ROLLING_WINDOW, MIN_PERIODS
from data.fetch import fetch_weekly_stats, fetch_schedules
from models.network import PropsNet


def load_model(prop_type):
    """Load a trained model and its scaler."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_best.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_scaler.pkl")

    ckpt = torch.load(ckpt_path, weights_only=False)
    model = PropsNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, ckpt["feature_cols"]


def get_player_features(player_name, prop_type, weekly, schedules, opponent_team):
    """Build a feature vector for a player's Super Bowl prediction."""
    cfg = PROP_TYPES[prop_type]

    # Find the player's recent games
    player_data = weekly[weekly["player_display_name"] == player_name].copy()
    if player_data.empty:
        print(f"  Player not found: {player_name}")
        return None

    player_data = player_data.sort_values(["season", "week"])

    # Use the last N games for rolling averages
    recent = player_data.tail(ROLLING_WINDOW)
    if len(recent) < MIN_PERIODS:
        print(f"  Not enough recent games for {player_name} ({len(recent)} games)")
        return None

    # Player rolling averages
    features = {}
    for col in cfg["player_features"]:
        features[f"{col}_roll{ROLLING_WINDOW}"] = recent[col].mean()

    # Opponent defensive rolling average
    opp_games = weekly[weekly["opponent_team"] == opponent_team].copy()
    opp_games = opp_games.sort_values(["season", "week"])
    opp_recent = opp_games.tail(ROLLING_WINDOW)
    def_total_per_game = opp_recent.groupby(["season", "week"])[cfg["defensive_stat"]].sum()
    features[f"opp_{cfg['defensive_stat']}_allowed_roll{ROLLING_WINDOW}"] = def_total_per_game.mean()

    # Super Bowl context: neutral site, ~14 days rest, postseason
    features["is_home"] = 0.0
    features["rest_days"] = 14.0
    # Super Bowl is typically week 22 in nfl_data_py
    features["week"] = 22.0
    features["is_postseason"] = 1.0

    return features


def predict_super_bowl(players_by_prop, sb_teams):
    """
    Generate Super Bowl predictions.

    Args:
        players_by_prop: dict mapping prop_type -> list of (player_name, team)
        sb_teams: tuple of (team1, team2) using nfl abbreviations (e.g. ("KC", "PHI"))
    """
    weekly = fetch_weekly_stats()
    schedules = fetch_schedules()

    team1, team2 = sb_teams

    print(f"\nSuper Bowl Predictions: {team1} vs {team2}")
    print("=" * 60)

    for prop_type, players in players_by_prop.items():
        model, scaler, feature_cols = load_model(prop_type)

        print(f"\n{prop_type.replace('_', ' ').title()}")
        print("-" * 40)

        for player_name, team in players:
            opponent = team2 if team == team1 else team1
            features = get_player_features(player_name, prop_type, weekly, schedules, opponent)

            if features is None:
                continue

            # Build feature vector in the correct column order
            feature_vector = np.array([[features[col] for col in feature_cols]])
            feature_vector = scaler.transform(feature_vector)

            with torch.no_grad():
                pred = model(torch.tensor(feature_vector, dtype=torch.float32)).item()

            pred = max(0, pred)  # yards can't be negative
            print(f"  {player_name:25s} → {pred:6.1f} yards")


def main():
    # Super Bowl LIX: Kansas City Chiefs vs Philadelphia Eagles
    sb_teams = ("KC", "PHI")

    players_by_prop = {
        "passing_yards": [
            ("Patrick Mahomes", "KC"),
            ("Jalen Hurts", "PHI"),
        ],
        "rushing_yards": [
            ("Isiah Pacheco", "KC"),
            ("Saquon Barkley", "PHI"),
            ("Patrick Mahomes", "KC"),
            ("Jalen Hurts", "PHI"),
        ],
        "receiving_yards": [
            ("Travis Kelce", "KC"),
            ("DeVonta Smith", "PHI"),
            ("A.J. Brown", "PHI"),
            ("Xavier Worthy", "KC"),
            ("Hollywood Brown", "KC"),
            ("Dallas Goedert", "PHI"),
        ],
    }

    predict_super_bowl(players_by_prop, sb_teams)


if __name__ == "__main__":
    main()
