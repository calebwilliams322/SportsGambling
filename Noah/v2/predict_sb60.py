"""
Super Bowl 60: Seattle Seahawks vs New England Patriots
V2 predictions with calibrated probabilities.

For each player, outputs:
  - Predicted yards
  - Per-prediction σ (uncertainty)
  - P(over) and P(under) for the actual betting line
  - What lines would give 65%, 70%, 80% probability
"""

import sys
import os
import torch
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v2.config import CHECKPOINT_DIR, PROP_TYPES, ROLLING_WINDOWS, MIN_PERIODS
from v2.data.fetch import fetch_weekly_stats, fetch_schedules
from v2.models.neural_net import PropsNetV2
from v2.models.gradient_boost import GradientBoostModel
from v2.models.ensemble import EnsembleModel
from v2.calibration.calibrator import PerPredictionCalibrator
from v2.data.features import (
    compute_opponent_defensive_rolling,
    compute_position_defensive_stats,
)

# Super Bowl 60 betting lines (same as v1)
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


def load_v2_model(prop_type):
    nn_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_nn.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_scaler.pkl")
    gbm_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_gbm.pkl")
    ensemble_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_ensemble.pkl")
    cal_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_calibrator.pkl")

    ckpt = torch.load(nn_path, weights_only=False)
    nn_model = PropsNetV2(input_dim=ckpt["input_dim"])
    nn_model.load_state_dict(ckpt["model_state_dict"])
    nn_model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    gbm_model = GradientBoostModel()
    gbm_model.load(gbm_path)

    with open(ensemble_path, "rb") as f:
        ens_data = pickle.load(f)
    ensemble = EnsembleModel(nn_weight=ens_data["nn_weight"], gbm_weight=ens_data["gbm_weight"])

    calibrator = PerPredictionCalibrator()
    calibrator.load(cal_path)

    return {
        "nn_model": nn_model,
        "gbm_model": gbm_model,
        "ensemble": ensemble,
        "scaler": scaler,
        "calibrator": calibrator,
        "feature_cols": ckpt["feature_cols"],
    }


def get_player_features(player_name, prop_type, weekly, opponent_team, feature_cols):
    """Build the v2 feature vector for a player's Super Bowl prediction."""
    cfg = PROP_TYPES[prop_type]

    player_data = weekly[weekly["player_display_name"] == player_name].copy()
    if player_data.empty:
        return None

    player_data = player_data.sort_values(["season", "week"])
    main_stat = cfg["player_features"][0]

    features = {}

    # Multi-window rolling means + std dev
    for col in cfg["player_features"]:
        for w in ROLLING_WINDOWS:
            recent = player_data.tail(w)
            if len(recent) >= MIN_PERIODS:
                features[f"{col}_roll{w}"] = recent[col].mean()
            else:
                features[f"{col}_roll{w}"] = np.nan

        max_w = max(ROLLING_WINDOWS)
        recent_max = player_data.tail(max_w)
        if len(recent_max) >= MIN_PERIODS:
            features[f"{col}_std{max_w}"] = recent_max[col].std()
        else:
            features[f"{col}_std{max_w}"] = np.nan

    # Season expanding average (use latest season data)
    latest_season = player_data["season"].max()
    season_data = player_data[player_data["season"] == latest_season]
    if len(season_data) >= MIN_PERIODS:
        features[f"{main_stat}_season_avg"] = season_data[main_stat].mean()
    else:
        features[f"{main_stat}_season_avg"] = np.nan

    # Efficiency features
    max_w = max(ROLLING_WINDOWS)
    for name, (numerator, denominator) in cfg.get("efficiency_features", {}).items():
        num_key = f"{numerator}_roll{max_w}"
        den_key = f"{denominator}_roll{max_w}"
        num_val = features.get(num_key, 0)
        den_val = features.get(den_key, 0)
        features[name] = num_val / den_val if den_val and den_val > 0 else np.nan

    # Trend (slope over last 5 games)
    trend_window = 5
    recent_trend = player_data.tail(trend_window)
    if len(recent_trend) >= trend_window:
        x = np.arange(trend_window, dtype=float)
        y = recent_trend[main_stat].values.astype(float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        slope = ((x * (y - y.mean())).sum()) / x_var if x_var > 0 else 0.0
        features[f"{main_stat}_trend"] = slope
    else:
        features[f"{main_stat}_trend"] = 0.0

    # Opponent defensive rolling (multi-window)
    opp_games = weekly[weekly["opponent_team"] == opponent_team].copy()
    opp_games = opp_games.sort_values(["season", "week"])
    for w in ROLLING_WINDOWS:
        opp_recent = opp_games.tail(w)
        if len(opp_recent) > 0:
            def_total_per_game = opp_recent.groupby(["season", "week"])[cfg["defensive_stat"]].sum()
            features[f"opp_{cfg['defensive_stat']}_allowed_roll{w}"] = def_total_per_game.mean()
        else:
            features[f"opp_{cfg['defensive_stat']}_allowed_roll{w}"] = np.nan

    # Position-specific defensive stats
    for pos in cfg["position_filter"]:
        pos_games = weekly[
            (weekly["opponent_team"] == opponent_team) & (weekly["position"] == pos)
        ].copy()
        pos_games = pos_games.sort_values(["season", "week"])
        max_w = max(ROLLING_WINDOWS)
        opp_recent = pos_games.tail(max_w)
        col_name = f"opp_{cfg['defensive_stat']}_to_{pos}_roll{max_w}"
        if col_name in feature_cols:
            if len(opp_recent) > 0:
                pos_total = opp_recent.groupby(["season", "week"])[cfg["defensive_stat"]].sum()
                features[col_name] = pos_total.mean()
            else:
                features[col_name] = np.nan

    # Context features — Super Bowl
    features["is_home"] = 0.0   # neutral site
    features["rest_days"] = 14.0
    features["week"] = 22.0
    features["is_postseason"] = 1.0

    return features


def main():
    weekly = fetch_weekly_stats()
    team1, team2 = SB_TEAMS

    print("=" * 85)
    print("SUPER BOWL 60: SEATTLE SEAHAWKS vs NEW ENGLAND PATRIOTS")
    print("V2 Model — Calibrated Probabilities with Per-Prediction Uncertainty")
    print("=" * 85)

    all_bets = []

    for prop_type, lines in LINES.items():
        v2 = load_v2_model(prop_type)

        print(f"\n{'─'*85}")
        print(f"  {prop_type.replace('_', ' ').upper()}")
        print(f"  (NN weight={v2['ensemble'].nn_weight:.2f}, GBM weight={v2['ensemble'].gbm_weight:.2f})")
        print(f"{'─'*85}")
        header = (
            f"  {'Player':<25} {'Line':>6} {'Pred':>7} {'σ':>5} "
            f"{'P(over)':>8} {'P(under)':>9} {'Bet':>10}"
        )
        print(header)
        print(f"  {'-'*78}")

        for player_name, team, line in lines:
            opponent = team2 if team == team1 else team1
            features = get_player_features(
                player_name, prop_type, weekly, opponent, v2["feature_cols"]
            )

            if features is None:
                print(f"  {player_name:<25} {'—':>6} {'N/A':>7}   (player not found)")
                continue

            # Build feature vector
            feature_vector_raw = np.array([[features.get(col, np.nan) for col in v2["feature_cols"]]])

            # Check for NaNs — fill with 0 for missing
            nan_mask = np.isnan(feature_vector_raw)
            if nan_mask.any():
                feature_vector_raw = np.nan_to_num(feature_vector_raw, nan=0.0)

            feature_vector_scaled = v2["scaler"].transform(feature_vector_raw)

            # NN prediction
            v2["nn_model"].eval()
            with torch.no_grad():
                nn_pred = v2["nn_model"](
                    torch.tensor(feature_vector_scaled, dtype=torch.float32)
                ).item()
            nn_pred = max(0, nn_pred)

            # GBM prediction
            gbm_pred = max(0, v2["gbm_model"].predict(feature_vector_raw)[0])

            # Ensemble prediction
            pred = v2["ensemble"].predict(
                np.array([nn_pred]), np.array([gbm_pred])
            )[0]

            # Per-prediction uncertainty
            sigma = v2["calibrator"].predict_sigma(feature_vector_raw)[0]

            # Calibrated probabilities
            p_over = v2["calibrator"].prob_over(pred, sigma, line)
            p_under = 1.0 - p_over

            # Determine bet direction
            if p_over > p_under:
                direction = "OVER"
                confidence = p_over
            else:
                direction = "UNDER"
                confidence = p_under

            if confidence >= 0.70:
                bet_label = f"★ {direction}"
            elif confidence >= 0.55:
                bet_label = f"  {direction}?"
            else:
                bet_label = "  SKIP"

            print(
                f"  {player_name:<25} {line:>6.1f} {pred:>7.1f} {sigma:>5.1f} "
                f"{p_over:>7.0%} {p_under:>8.0%} {bet_label:>10}"
            )

            all_bets.append({
                "player": player_name,
                "prop": prop_type,
                "line": line,
                "pred": pred,
                "sigma": sigma,
                "p_over": p_over,
                "p_under": p_under,
                "direction": direction,
                "confidence": confidence,
            })

    # ── Summary of recommended bets ───────────────────────────────────
    strong_bets = [b for b in all_bets if b["confidence"] >= 0.70]
    lean_bets = [b for b in all_bets if 0.55 <= b["confidence"] < 0.70]

    print(f"\n{'='*85}")
    print("RECOMMENDED BETS (70%+ Calibrated Probability)")
    print(f"{'='*85}")
    if strong_bets:
        for b in sorted(strong_bets, key=lambda x: -x["confidence"]):
            prop_label = b["prop"].replace("_", " ").title()
            print(f"  ★ {b['player']} {prop_label} {b['direction']} {b['line']}")
            print(f"    Predicted: {b['pred']:.1f} ± {b['sigma']:.1f} | "
                  f"P(over): {b['p_over']:.0%} | P(under): {b['p_under']:.0%}")

            # Show what lines would give various probabilities
            calibrator = load_v2_model(b["prop"])["calibrator"]
            print(f"    Lines for target probabilities:")
            for target in [0.65, 0.70, 0.80]:
                line_over = calibrator.find_line_for_prob(b["pred"], b["sigma"], target)
                line_under = calibrator.find_line_for_prob(b["pred"], b["sigma"], 1.0 - target)
                print(f"      {target:.0%} OVER at line {line_over:.1f} | "
                      f"{target:.0%} UNDER at line {line_under:.1f}")
            print()
    else:
        print("  No bets meet the 70% calibrated probability threshold.")

    if lean_bets:
        print(f"\n{'─'*85}")
        print("LEANS (55-69% — proceed with caution)")
        print(f"{'─'*85}")
        for b in sorted(lean_bets, key=lambda x: -x["confidence"]):
            prop_label = b["prop"].replace("_", " ").title()
            print(f"  ? {b['player']} {prop_label} {b['direction']} {b['line']} "
                  f"(Pred: {b['pred']:.1f}±{b['sigma']:.1f}, "
                  f"P({b['direction'].lower()}): {b['confidence']:.0%})")


if __name__ == "__main__":
    main()
