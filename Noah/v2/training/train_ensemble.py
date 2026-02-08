"""
Full training pipeline orchestrator.

1. Build features
2. Run cross-validation to get out-of-sample predictions
3. Fit calibrator on CV predictions
4. Train final NN + GBM on 2015-2023
5. Optimize ensemble weights on 2024 holdout
6. Evaluate on 2025 test set
7. Save all artifacts
"""

import os
import pickle
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from v2.config import (
    CHECKPOINT_DIR, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, PROP_TYPES,
)
from v2.data.features import build_features
from v2.training.train_nn import train_nn, predict_nn
from v2.training.train_gbm import train_gbm, predict_gbm
from v2.training.cross_val import leave_one_season_out_cv
from v2.models.ensemble import EnsembleModel
from v2.calibration.calibrator import PerPredictionCalibrator


def train_prop_type(prop_type):
    """Train the full v2 pipeline for one prop type."""
    print(f"\n{'='*70}")
    print(f"V2 TRAINING: {prop_type}")
    print(f"{'='*70}")

    # ── Step 1: Build features ────────────────────────────────────────
    print("\n[1/7] Building features...")
    df, feature_cols = build_features(prop_type)
    print(f"  Total rows: {len(df)}, features: {len(feature_cols)}")

    # ── Step 2: Cross-validation ──────────────────────────────────────
    print("\n[2/7] Running leave-one-season-out cross-validation...")
    cv_features, cv_predictions, cv_actuals, cv_seasons = leave_one_season_out_cv(
        df, feature_cols, prop_type
    )
    cv_mae = np.mean(np.abs(cv_predictions - cv_actuals))
    print(f"  Overall CV MAE: {cv_mae:.1f}")

    # ── Step 3: Fit calibrator ────────────────────────────────────────
    print("\n[3/7] Fitting per-prediction calibrator...")
    calibrator = PerPredictionCalibrator()
    calibrator.fit(cv_features, cv_predictions, cv_actuals)
    print("  Calibrator fitted on CV data")

    # ── Step 4: Train final models on 2015-2023 ───────────────────────
    print("\n[4/7] Training final models on 2015-2023...")
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[prop_type].values
    X_val_raw = val_df[feature_cols].values
    y_val = val_df[prop_type].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[prop_type].values

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Train NN
    print("\n  Training Neural Network...")
    nn_model, nn_val_loss = train_nn(
        X_train_scaled, y_train, X_val_scaled, y_val,
        input_dim=len(feature_cols),
    )
    nn_train_preds = predict_nn(nn_model, X_train_scaled)
    nn_val_preds = predict_nn(nn_model, X_val_scaled)
    nn_test_preds = predict_nn(nn_model, X_test_scaled)
    nn_test_mae = np.mean(np.abs(nn_test_preds - y_test))
    print(f"  NN test MAE: {nn_test_mae:.1f}")

    # Train GBM
    print("\n  Training LightGBM...")
    gbm_model, gbm_val_mae = train_gbm(X_train_raw, y_train, X_val_raw, y_val)
    gbm_val_preds = predict_gbm(gbm_model, X_val_raw)
    gbm_test_preds = predict_gbm(gbm_model, X_test_raw)
    gbm_test_mae = np.mean(np.abs(gbm_test_preds - y_test))
    print(f"  GBM test MAE: {gbm_test_mae:.1f}")

    # ── Step 5: Optimize ensemble weights on 2024 ─────────────────────
    print("\n[5/7] Optimizing ensemble weights on 2024 holdout...")
    ensemble = EnsembleModel.optimize_weights(nn_val_preds, gbm_val_preds, y_val)
    print(f"  Optimal weights: NN={ensemble.nn_weight:.2f}, GBM={ensemble.gbm_weight:.2f}")

    # ── Step 6: Evaluate on 2025 test set ─────────────────────────────
    print("\n[6/7] Evaluating on 2025 test set...")
    ensemble_test_preds = ensemble.predict(nn_test_preds, gbm_test_preds)
    ensemble_test_mae = np.mean(np.abs(ensemble_test_preds - y_test))
    baseline_mae = np.mean(np.abs(y_test - y_test.mean()))

    print(f"  NN test MAE:       {nn_test_mae:.1f}")
    print(f"  GBM test MAE:      {gbm_test_mae:.1f}")
    print(f"  Ensemble test MAE: {ensemble_test_mae:.1f}")
    print(f"  Baseline MAE:      {baseline_mae:.1f}")
    print(f"  Improvement:       {baseline_mae - ensemble_test_mae:.1f} yards")

    # Calibration quality on test set
    test_sigmas = calibrator.predict_sigma(X_test_raw)
    # Check calibration at a few probability thresholds
    print("\n  Calibration check (test set):")
    for target_p in [0.3, 0.5, 0.7]:
        lines_at_p = np.array([
            calibrator.find_line_for_prob(pred, sig, target_p)
            for pred, sig in zip(ensemble_test_preds, test_sigmas)
        ])
        actual_over_rate = np.mean(y_test > lines_at_p)
        print(f"    P(over)={target_p:.0%} → actual={actual_over_rate:.0%}")

    # ── Step 7: Save artifacts ────────────────────────────────────────
    print("\n[7/7] Saving artifacts...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # NN checkpoint
    torch.save({
        "model_state_dict": nn_model.state_dict(),
        "feature_cols": feature_cols,
        "prop_type": prop_type,
        "input_dim": len(feature_cols),
    }, os.path.join(CHECKPOINT_DIR, f"{prop_type}_nn.pt"))

    # Scaler
    with open(os.path.join(CHECKPOINT_DIR, f"{prop_type}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # GBM
    gbm_model.save(os.path.join(CHECKPOINT_DIR, f"{prop_type}_gbm.pkl"))

    # Ensemble weights
    with open(os.path.join(CHECKPOINT_DIR, f"{prop_type}_ensemble.pkl"), "wb") as f:
        pickle.dump({
            "nn_weight": ensemble.nn_weight,
            "gbm_weight": ensemble.gbm_weight,
        }, f)

    # Calibrator
    calibrator.save(os.path.join(CHECKPOINT_DIR, f"{prop_type}_calibrator.pkl"))

    print(f"  All artifacts saved to {CHECKPOINT_DIR}")

    return {
        "prop_type": prop_type,
        "nn_mae": nn_test_mae,
        "gbm_mae": gbm_test_mae,
        "ensemble_mae": ensemble_test_mae,
        "baseline_mae": baseline_mae,
        "nn_weight": ensemble.nn_weight,
        "gbm_weight": ensemble.gbm_weight,
        "n_features": len(feature_cols),
    }
