"""Side-by-side comparison of v1 and v2 models on the 2025 test set."""

import sys
import os
import torch
import pickle
import numpy as np
from scipy.stats import norm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from v2.config import (
    PROP_TYPES, CHECKPOINT_DIR, TEST_SEASONS, V1_CHECKPOINT_DIR,
)
from v2.data.features import build_features as build_features_v2
from v2.training.train_nn import predict_nn
from v2.training.train_gbm import predict_gbm
from v2.models.neural_net import PropsNetV2
from v2.models.gradient_boost import GradientBoostModel
from v2.models.ensemble import EnsembleModel
from v2.calibration.calibrator import PerPredictionCalibrator
from v2.evaluate.metrics import expected_calibration_error, brier_score, reliability_analysis

# V1 imports
from config import PROP_TYPES as V1_PROP_TYPES
from data.features import build_features as build_features_v1
from models.network import PropsNet


def load_v1_model(prop_type):
    ckpt_path = os.path.join(V1_CHECKPOINT_DIR, f"{prop_type}_best.pt")
    scaler_path = os.path.join(V1_CHECKPOINT_DIR, f"{prop_type}_scaler.pkl")

    if not os.path.exists(ckpt_path):
        return None, None, None

    ckpt = torch.load(ckpt_path, weights_only=False)
    model = PropsNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, ckpt["feature_cols"]


def load_v2_model(prop_type):
    nn_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_nn.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_scaler.pkl")
    gbm_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_gbm.pkl")
    ensemble_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_ensemble.pkl")
    cal_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_calibrator.pkl")

    if not os.path.exists(nn_path):
        return None

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


def main():
    print("=" * 75)
    print("V1 vs V2 MODEL COMPARISON — 2025 Test Set")
    print("=" * 75)

    # V1 error stds (from v1 predict_sb60.py)
    v1_error_std = {
        "passing_yards": 85.5,
        "rushing_yards": 26.1,
        "receiving_yards": 25.6,
    }

    for prop_type in PROP_TYPES:
        print(f"\n{'─'*75}")
        print(f"  {prop_type.replace('_', ' ').upper()}")
        print(f"{'─'*75}")

        # ── V1 ────────────────────────────────────────────────────────
        v1_model, v1_scaler, v1_feature_cols = load_v1_model(prop_type)
        if v1_model is None:
            print("  V1 model not found — skipping v1")
            v1_mae = None
        else:
            v1_df, _ = build_features_v1(prop_type)
            v1_test = v1_df[v1_df["season"].isin(TEST_SEASONS)]
            X_v1 = v1_scaler.transform(v1_test[v1_feature_cols].values)
            y_test = v1_test[prop_type].values

            with torch.no_grad():
                v1_preds = v1_model(torch.tensor(X_v1, dtype=torch.float32)).numpy()
            v1_preds = np.maximum(v1_preds, 0)
            v1_mae = np.mean(np.abs(v1_preds - y_test))

            # V1 "calibration" — using global sigma
            std = v1_error_std[prop_type]
            # Simulate: for each prediction, pretend the line = actual outcome
            # and see if v1's directional confidence is calibrated
            v1_edges = np.abs(v1_preds - y_test)
            v1_probs = norm.cdf(v1_edges / std)
            v1_correct = ((v1_preds > y_test) == (v1_preds > y_test)).astype(float)
            # More meaningful: directional accuracy at various confidence levels
            print(f"\n  V1: MAE = {v1_mae:.1f}")
            print(f"  V1: {len(v1_feature_cols)} features")

        # ── V2 ────────────────────────────────────────────────────────
        v2 = load_v2_model(prop_type)
        if v2 is None:
            print("  V2 model not found — run v2/train.py first")
            continue

        v2_df, v2_feature_cols = build_features_v2(prop_type)
        v2_test = v2_df[v2_df["season"].isin(TEST_SEASONS)]
        X_v2_raw = v2_test[v2_feature_cols].values
        X_v2_scaled = v2["scaler"].transform(X_v2_raw)
        y_test_v2 = v2_test[prop_type].values

        nn_preds = predict_nn(v2["nn_model"], X_v2_scaled)
        gbm_preds = predict_gbm(v2["gbm_model"], X_v2_raw)
        v2_preds = v2["ensemble"].predict(nn_preds, gbm_preds)
        v2_mae = np.mean(np.abs(v2_preds - y_test_v2))

        print(f"\n  V2: MAE = {v2_mae:.1f}")
        print(f"  V2: {len(v2_feature_cols)} features")
        print(f"  V2: NN weight={v2['ensemble'].nn_weight:.2f}, GBM weight={v2['ensemble'].gbm_weight:.2f}")

        if v1_mae is not None:
            improvement = v1_mae - v2_mae
            print(f"\n  MAE improvement: {improvement:+.1f} yards ({'better' if improvement > 0 else 'worse'})")

        # V2 calibration analysis
        sigmas = v2["calibrator"].predict_sigma(X_v2_raw)

        # Generate calibration data: for random lines around predictions
        np.random.seed(42)
        n_test = len(v2_preds)
        offsets = np.random.uniform(-1.5, 1.5, size=n_test) * sigmas
        synthetic_lines = v2_preds + offsets
        predicted_probs = v2["calibrator"].prob_over_batch(
            v2_preds, sigmas, synthetic_lines, calibrate=True
        )
        actual_outcomes = (y_test_v2 > synthetic_lines).astype(float)

        print(f"\n  V2 Calibration Analysis:")
        ece = reliability_analysis(predicted_probs, actual_outcomes)
        bs = brier_score(predicted_probs, actual_outcomes)
        print(f"  Brier Score: {bs:.3f}")

    print(f"\n{'='*75}")
    print("Done.")


if __name__ == "__main__":
    main()
