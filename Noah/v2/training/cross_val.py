"""Leave-one-season-out cross-validation for generating calibration data."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from v2.config import CV_SEASONS
from v2.training.train_nn import train_nn, predict_nn
from v2.training.train_gbm import train_gbm, predict_gbm
from v2.models.ensemble import EnsembleModel


def leave_one_season_out_cv(df, feature_cols, prop_type, verbose=True):
    """
    Run leave-one-season-out CV across CV_SEASONS.
    Returns arrays of (features, predictions, actuals, seasons) for all
    out-of-sample folds — this becomes calibration training data.
    """
    all_features = []
    all_predictions = []
    all_actuals = []
    all_seasons = []

    for hold_season in CV_SEASONS:
        if verbose:
            print(f"\n  CV fold: hold out {hold_season}")

        train_mask = (df["season"] < hold_season) & (df["season"] >= 2015)
        val_mask = df["season"] == hold_season

        train_df = df[train_mask]
        val_df = df[val_mask]

        if len(train_df) < 100 or len(val_df) < 10:
            if verbose:
                print(f"    Skipping — train={len(train_df)}, val={len(val_df)}")
            continue

        X_train_raw = train_df[feature_cols].values
        y_train = train_df[prop_type].values
        X_val_raw = val_df[feature_cols].values
        y_val = val_df[prop_type].values

        # Scale for NN (GBM uses raw)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        # Use a portion of training data as NN validation for early stopping
        n_train = len(X_train_scaled)
        split_idx = int(n_train * 0.85)
        X_nn_train = X_train_scaled[:split_idx]
        y_nn_train = y_train[:split_idx]
        X_nn_val = X_train_scaled[split_idx:]
        y_nn_val = y_train[split_idx:]

        # Train NN
        nn_model, _ = train_nn(
            X_nn_train, y_nn_train, X_nn_val, y_nn_val,
            input_dim=len(feature_cols), verbose=False,
        )
        nn_preds = predict_nn(nn_model, X_val_scaled)

        # Train GBM
        gbm_model, _ = train_gbm(
            X_train_raw[:split_idx], y_train[:split_idx],
            X_train_raw[split_idx:], y_train[split_idx:],
            verbose=False,
        )
        gbm_preds = predict_gbm(gbm_model, X_val_raw)

        # Ensemble (use default weights for CV)
        ensemble = EnsembleModel()
        ensemble_preds = ensemble.predict(nn_preds, gbm_preds)

        mae = np.mean(np.abs(ensemble_preds - y_val))
        if verbose:
            print(f"    Fold MAE: {mae:.1f} (n={len(y_val)})")

        all_features.append(X_val_raw)
        all_predictions.append(ensemble_preds)
        all_actuals.append(y_val)
        all_seasons.append(np.full(len(y_val), hold_season))

    return (
        np.concatenate(all_features),
        np.concatenate(all_predictions),
        np.concatenate(all_actuals),
        np.concatenate(all_seasons),
    )
