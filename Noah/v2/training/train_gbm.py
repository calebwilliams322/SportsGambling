"""LightGBM training with early stopping."""

import numpy as np

from v2.models.gradient_boost import GradientBoostModel


def train_gbm(X_train, y_train, X_val, y_val, verbose=True):
    """
    Train LightGBM and return (model, val_mae).
    GBM doesn't need scaled features — it's tree-based.
    """
    model = GradientBoostModel()
    model.fit(X_train, y_train, X_val, y_val)

    val_preds = model.predict(X_val)
    val_mae = np.mean(np.abs(val_preds - y_val))

    if verbose:
        print(f"    GBM val MAE: {val_mae:.1f}")

    return model, val_mae


def predict_gbm(model, X):
    """Run inference."""
    preds = model.predict(X)
    return np.maximum(preds, 0)
