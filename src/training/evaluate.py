"""
Model evaluation — compute metrics on test set and compare to baselines.
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model, data_loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run predictions on a DataLoader. Returns (predictions, actuals)."""
    model.eval()
    all_preds = []
    all_actuals = []

    for X, y in data_loader:
        X = X.to(device)
        preds = model(X).cpu().numpy()
        all_preds.append(preds)
        all_actuals.append(y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_actuals)


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute regression metrics."""
    errors = predictions - actuals
    abs_errors = np.abs(errors)

    return {
        "mae": float(np.mean(abs_errors)),
        "median_ae": float(np.median(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_prediction": float(np.mean(predictions)),
        "mean_actual": float(np.mean(actuals)),
        "std_prediction": float(np.std(predictions)),
        "std_actual": float(np.std(actuals)),
        "n_samples": len(actuals),
    }


def evaluate_model(model, test_loader, device="cpu", stat_name="stat") -> dict:
    """
    Full evaluation: compute metrics and print a summary.
    """
    predictions, actuals = predict(model, test_loader, device)
    metrics = compute_metrics(predictions, actuals)

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results: {stat_name}")
    print(f"{'=' * 50}")
    print(f"  Samples:          {metrics['n_samples']}")
    print(f"  MAE:              {metrics['mae']:.2f}")
    print(f"  Median AE:        {metrics['median_ae']:.2f}")
    print(f"  RMSE:             {metrics['rmse']:.2f}")
    print(f"  Mean Prediction:  {metrics['mean_prediction']:.2f}")
    print(f"  Mean Actual:      {metrics['mean_actual']:.2f}")
    print(f"{'=' * 50}")

    return {
        "metrics": metrics,
        "predictions": predictions,
        "actuals": actuals,
    }


def compare_to_baseline(predictions, actuals, stat_name="stat"):
    """
    Compare model to a naive baseline (predicting the mean).
    Shows how much better the model is vs. just guessing the average.
    """
    baseline_pred = np.full_like(actuals, np.mean(actuals))
    baseline_mae = np.mean(np.abs(baseline_pred - actuals))
    model_mae = np.mean(np.abs(predictions - actuals))

    improvement = (baseline_mae - model_mae) / baseline_mae * 100

    print(f"\nBaseline Comparison ({stat_name}):")
    print(f"  Baseline MAE (predict mean): {baseline_mae:.2f}")
    print(f"  Model MAE:                   {model_mae:.2f}")
    print(f"  Improvement:                 {improvement:.1f}%")

    return {"baseline_mae": baseline_mae, "model_mae": model_mae,
            "improvement_pct": improvement}
