"""Evaluation metrics: Expected Calibration Error (ECE), Brier score, reliability analysis."""

import numpy as np


def expected_calibration_error(predicted_probs, actual_outcomes, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Bins predictions by predicted probability, checks actual win rates match.
    Target: < 5%.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if i == n_bins - 1:
            mask = (predicted_probs >= lo) & (predicted_probs <= hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_predicted = predicted_probs[mask].mean()
        avg_actual = actual_outcomes[mask].mean()
        gap = abs(avg_predicted - avg_actual)
        ece += gap * (n_in_bin / len(predicted_probs))

        bin_details.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(n_in_bin),
            "avg_predicted": float(avg_predicted),
            "avg_actual": float(avg_actual),
            "gap": float(gap),
        })

    return float(ece), bin_details


def brier_score(predicted_probs, actual_outcomes):
    """Brier score: mean squared error of probability predictions. Lower is better."""
    return float(np.mean((predicted_probs - actual_outcomes) ** 2))


def reliability_analysis(predicted_probs, actual_outcomes, n_bins=10):
    """Print a reliability table showing predicted vs actual win rates per bin."""
    ece, bins = expected_calibration_error(predicted_probs, actual_outcomes, n_bins)
    print(f"\n  {'Bin':<12} {'N':>6} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
    print(f"  {'-'*48}")
    for b in bins:
        print(f"  {b['bin']:<12} {b['n']:>6} {b['avg_predicted']:>9.1%} {b['avg_actual']:>9.1%} {b['gap']:>7.1%}")
    print(f"\n  ECE: {ece:.3f} (target: < 0.05)")
    return ece
