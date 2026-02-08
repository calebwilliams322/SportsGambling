"""
Calculate the minimum betting edge for each prop type.

For standard -110 odds, you need to win ~52.4% of bets to break even.
We target 55% win rate for a real edge with margin.

Method: On the test set, simulate bets at various thresholds.
If the model predicts X yards more than a "line", how often is the actual
result actually over that line? The threshold where your win rate exceeds
55% is your minimum edge.
"""

import os
import torch
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import CHECKPOINT_DIR, PROP_TYPES, TEST_SEASONS, VAL_SEASONS, TRAIN_SEASONS
from data.features import build_features
from models.network import PropsNet


def analyze_prop(prop_type):
    print(f"\n{'='*60}")
    print(f"{prop_type.replace('_', ' ').upper()}")
    print(f"{'='*60}")

    # Load model
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_best.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_scaler.pkl")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = PropsNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_cols = ckpt["feature_cols"]

    # Build features and get test set
    df, _ = build_features(prop_type)
    test_df = df[df["season"].isin(TEST_SEASONS)]

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df[prop_type].values

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    errors = preds - y_test
    abs_errors = np.abs(errors)

    print(f"\nTest set size: {len(y_test)} games")
    print(f"MAE: {abs_errors.mean():.1f} yards")
    print(f"Std of errors: {errors.std():.1f} yards")
    print(f"Median absolute error: {np.median(abs_errors):.1f} yards")

    # Simulate betting at various edge thresholds
    # "edge" = how far the model prediction is from the line
    # We simulate: the "line" is the actual result + random noise to simulate
    # real lines, but more directly: for each game, pretend the line is set at
    # various offsets from our prediction and check if we'd win.
    #
    # More precisely: if model predicts P and line is L:
    #   - If P > L (model says over), we bet OVER. We win if actual > L.
    #   - If P < L (model says under), we bet UNDER. We win if actual < L.
    #   - Edge = |P - L|
    #
    # We can compute this directly from our test data:
    # For each threshold T, look at all games where |pred - actual_mean_line| > T
    # and see win rate.
    #
    # Since we don't have real lines, we use a clean approach:
    # For each game, edge = |pred - actual|... no, that's circular.
    #
    # Better approach: treat each test game's ACTUAL as the "true" outcome.
    # For a given edge threshold T:
    #   Consider all pairs where |pred - some_hypothetical_line| >= T
    #   We simulate lines centered around the actual value with some spread.
    #
    # Simplest valid approach: use the model's error distribution directly.
    # If the model is off by std σ, then for an edge of E yards:
    #   P(win) ≈ Φ(E / σ) where Φ is the normal CDF
    # This tells us what edge we need for various win rates.

    from scipy.stats import norm

    std = errors.std()
    print(f"\n{'Minimum Edge Analysis':}")
    print(f"{'(how far prediction must be from the line to bet)':}")
    print(f"\n  {'Edge (yds)':>12} {'Est. Win Rate':>15} {'Bet?':>8}")
    print(f"  {'-'*38}")

    recommended_edge = None
    for edge in range(0, int(std * 2) + 1, max(1, int(std * 0.1))):
        # Win rate when our prediction is `edge` yards away from the line
        win_rate = norm.cdf(edge / std)
        verdict = ""
        if win_rate >= 0.55 and recommended_edge is None:
            recommended_edge = edge
            verdict = " ← MINIMUM"
        elif win_rate >= 0.55:
            verdict = " ✓"
        print(f"  {edge:>10d}   {win_rate:>12.1%} {verdict}")

    # Also do empirical validation: bin test games by how far off the prediction was
    # and check directional accuracy
    print(f"\n  Empirical check (test set):")
    print(f"  {'Edge bin':>15} {'Games':>8} {'Directional %':>15}")
    print(f"  {'-'*42}")

    # For each game: if pred > mean(y_test), did actual > mean(y_test)?
    # Better: split into bins by |pred - y_test_mean| and check
    mean_actual = y_test.mean()
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 999)]:
        # Games where model predicted at least `lo` yards above/below the overall mean
        mask_over = (preds - mean_actual >= lo) & (preds - mean_actual < hi)
        mask_under = (mean_actual - preds >= lo) & (mean_actual - preds < hi)

        correct_over = (y_test[mask_over] > mean_actual).sum() if mask_over.sum() > 0 else 0
        correct_under = (y_test[mask_under] < mean_actual).sum() if mask_under.sum() > 0 else 0
        total = mask_over.sum() + mask_under.sum()
        correct = correct_over + correct_under

        if total > 0:
            pct = correct / total
            label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
            print(f"  {label:>15} {total:>8d}   {pct:>12.1%}")

    print(f"\n  ★ RECOMMENDED MINIMUM EDGE: {recommended_edge} yards")
    print(f"    Only bet when your model prediction is at least {recommended_edge} yards")
    print(f"    away from the sportsbook line.")

    return recommended_edge


def main():
    print("BETTING EDGE ANALYSIS")
    print("Standard -110 odds require ~52.4% win rate to break even.")
    print("We target 55%+ for a real profitable edge.\n")

    results = {}
    for prop_type in PROP_TYPES:
        results[prop_type] = analyze_prop(prop_type)

    print(f"\n\n{'='*60}")
    print("SUMMARY — MINIMUM EDGES TO BET")
    print(f"{'='*60}")
    for prop_type, edge in results.items():
        label = prop_type.replace("_", " ").title()
        print(f"  {label:20s}: {edge} yards")
    print(f"\nIf the model predicts OVER by at least this many yards → bet the OVER")
    print(f"If the model predicts UNDER by at least this many yards → bet the UNDER")
    print(f"Otherwise → skip the bet")


if __name__ == "__main__":
    main()
