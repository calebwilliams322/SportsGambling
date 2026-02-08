"""
Super Bowl 60: Seattle Seahawks vs New England Patriots
Compare model predictions against actual betting lines.
Flag bets where confidence >= 80%.
"""

import os
import torch
import pickle
import numpy as np
from scipy.stats import norm

from config import CHECKPOINT_DIR, PROP_TYPES, ROLLING_WINDOW, MIN_PERIODS
from data.fetch import fetch_weekly_stats, fetch_schedules
from models.network import PropsNet


# Super Bowl 60 betting lines
LINES = {
    "passing_yards": [
        ("Sam Darnold", "SEA", 230.5),
        ("Drake Maye", "NE", 219.5),
    ],
    "rushing_yards": [
        ("Kenneth Walker III", "SEA", 72.5),
        ("Rhamondre Stevenson", "NE", 48.5),
        ("Drake Maye", "NE", 35.5),
        ("TreVeyon Henderson", "NE", 17.5),
        ("Sam Darnold", "SEA", 5.5),
    ],
    "receiving_yards": [
        ("Jaxon Smith-Njigba", "SEA", 93.5),
        ("Stefon Diggs", "SEA", 43.5),
        ("Hunter Henry", "NE", 39.5),
        ("Cooper Kupp", "SEA", 33.5),
        ("Kayshon Boutte", "NE", 29.5),
        ("A.J. Barner", "SEA", 25.5),
        ("Mack Hollins", "SEA", 24.5),
        ("Rhamondre Stevenson", "NE", 24.5),
        ("Kenneth Walker III", "SEA", 21.5),
        ("Rashid Shaheed", "SEA", 21.5),
        ("Demario Douglas", "NE", 10.5),
    ],
}

SB_TEAMS = ("SEA", "NE")

# Model error std devs (from training analysis)
# Used to calculate confidence: P(win) = Phi(edge / std)
ERROR_STD = {
    "passing_yards": 85.5,
    "rushing_yards": 26.1,
    "receiving_yards": 25.6,
}


def load_model(prop_type):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_best.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_scaler.pkl")

    ckpt = torch.load(ckpt_path, weights_only=False)
    model = PropsNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, ckpt["feature_cols"]


def get_player_features(player_name, prop_type, weekly, opponent_team):
    cfg = PROP_TYPES[prop_type]

    player_data = weekly[weekly["player_display_name"] == player_name].copy()
    if player_data.empty:
        return None

    player_data = player_data.sort_values(["season", "week"])
    recent = player_data.tail(ROLLING_WINDOW)
    if len(recent) < MIN_PERIODS:
        return None

    features = {}
    for col in cfg["player_features"]:
        features[f"{col}_roll{ROLLING_WINDOW}"] = recent[col].mean()

    opp_games = weekly[weekly["opponent_team"] == opponent_team].copy()
    opp_games = opp_games.sort_values(["season", "week"])
    opp_recent = opp_games.tail(ROLLING_WINDOW)
    def_total_per_game = opp_recent.groupby(["season", "week"])[cfg["defensive_stat"]].sum()
    features[f"opp_{cfg['defensive_stat']}_allowed_roll{ROLLING_WINDOW}"] = def_total_per_game.mean()

    features["is_home"] = 0.0   # neutral site
    features["rest_days"] = 14.0
    features["week"] = 22.0
    features["is_postseason"] = 1.0

    return features


def main():
    weekly = fetch_weekly_stats()

    team1, team2 = SB_TEAMS

    print("=" * 75)
    print("SUPER BOWL 60: SEATTLE SEAHAWKS vs NEW ENGLAND PATRIOTS")
    print("Model Predictions vs Betting Lines — 80% Confidence Threshold")
    print("=" * 75)

    all_bets = []

    for prop_type, lines in LINES.items():
        model, scaler, feature_cols = load_model(prop_type)
        std = ERROR_STD[prop_type]

        print(f"\n{'─'*75}")
        print(f"  {prop_type.replace('_', ' ').upper()}")
        print(f"{'─'*75}")
        print(f"  {'Player':<25} {'Line':>6} {'Pred':>7} {'Edge':>7} {'Conf':>7} {'Bet':>10}")
        print(f"  {'-'*63}")

        for player_name, team, line in lines:
            opponent = team2 if team == team1 else team1
            features = get_player_features(player_name, prop_type, weekly, opponent)

            if features is None:
                print(f"  {player_name:<25} {'—':>6} {'N/A':>7}   (player not found in data)")
                continue

            feature_vector = np.array([[features[col] for col in feature_cols]])
            feature_vector = scaler.transform(feature_vector)

            with torch.no_grad():
                pred = model(torch.tensor(feature_vector, dtype=torch.float32)).item()
            pred = max(0, pred)

            edge = pred - line
            abs_edge = abs(edge)
            confidence = norm.cdf(abs_edge / std)

            if edge > 0:
                direction = "OVER"
            else:
                direction = "UNDER"

            if confidence >= 0.80:
                bet_label = f"★ {direction}"
            elif confidence >= 0.55:
                bet_label = f"  {direction}?"
            else:
                bet_label = "  SKIP"

            print(f"  {player_name:<25} {line:>6.1f} {pred:>7.1f} {edge:>+7.1f} {confidence:>6.0%} {bet_label:>10}")

            all_bets.append({
                "player": player_name,
                "prop": prop_type,
                "line": line,
                "pred": pred,
                "edge": edge,
                "confidence": confidence,
                "direction": direction,
            })

    # Summary of recommended bets
    strong_bets = [b for b in all_bets if b["confidence"] >= 0.80]
    lean_bets = [b for b in all_bets if 0.55 <= b["confidence"] < 0.80]

    print(f"\n{'='*75}")
    print("RECOMMENDED BETS (80%+ Confidence)")
    print(f"{'='*75}")
    if strong_bets:
        for b in sorted(strong_bets, key=lambda x: -x["confidence"]):
            prop_label = b["prop"].replace("_", " ").title()
            print(f"  ★ {b['player']} {prop_label} {b['direction']} {b['line']}")
            print(f"    Predicted: {b['pred']:.1f} | Edge: {b['edge']:+.1f} yds | Confidence: {b['confidence']:.0%}")
            print()
    else:
        print("  No bets meet the 80% confidence threshold.")

    if lean_bets:
        print(f"\n{'─'*75}")
        print("LEANS (55-79% Confidence — proceed with caution)")
        print(f"{'─'*75}")
        for b in sorted(lean_bets, key=lambda x: -x["confidence"]):
            prop_label = b["prop"].replace("_", " ").title()
            print(f"  ? {b['player']} {prop_label} {b['direction']} {b['line']} "
                  f"(Pred: {b['pred']:.1f}, Conf: {b['confidence']:.0%})")


if __name__ == "__main__":
    main()
