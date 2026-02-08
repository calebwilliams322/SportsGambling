"""
Per-prediction uncertainty estimation and calibrated probabilities.

Instead of one global σ, a secondary LightGBM model learns to predict the
absolute error of the main ensemble for each specific prediction. This captures
the fact that volatile players have higher uncertainty than consistent ones.

Pipeline:
  1. Train uncertainty model on CV out-of-sample |errors|
  2. At prediction time: ensemble predicts Y, uncertainty model predicts σ
  3. For any line L: P(over L) = 1 - Φ((L - Y) / σ)
  4. Optional isotonic regression correction if normal assumption doesn't hold
"""

import os
import pickle
import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb


class PerPredictionCalibrator:
    def __init__(self):
        self.uncertainty_model = None
        self.isotonic = None
        self.min_sigma = 1.0  # floor to avoid division by zero

    def fit(self, X_features, predictions, actuals):
        """
        Fit the uncertainty model and optional isotonic correction.

        Args:
            X_features: feature matrix (same features used by the ensemble)
            predictions: ensemble out-of-sample predictions
            actuals: actual outcomes
        """
        abs_errors = np.abs(predictions - actuals)

        # Step 1: Train uncertainty model to predict |error| from features
        self.uncertainty_model = lgb.LGBMRegressor(
            objective="regression",
            metric="mae",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=500,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
        )
        self.uncertainty_model.fit(X_features, abs_errors)

        # Step 2: Build isotonic correction on the CV data
        predicted_sigma = self._predict_sigma(X_features)
        errors = predictions - actuals

        # Generate a grid of lines around each prediction to build calibration data
        raw_probs = []
        actual_outcomes = []
        for i in range(len(predictions)):
            # Simulate checking at the actual outcome as the "line"
            # P(over actual) should be ~50% if model is perfect
            # We use multiple synthetic lines around the prediction
            for offset in np.linspace(-2, 2, 5):
                line = predictions[i] + offset * predicted_sigma[i]
                raw_p = self._raw_prob_over(predictions[i], predicted_sigma[i], line)
                actual_over = float(actuals[i] > line)
                raw_probs.append(raw_p)
                actual_outcomes.append(actual_over)

        raw_probs = np.array(raw_probs)
        actual_outcomes = np.array(actual_outcomes)

        # Fit isotonic regression: raw_prob → actual win rate
        self.isotonic = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        self.isotonic.fit(raw_probs, actual_outcomes)

    def predict_sigma(self, X_features):
        """Predict per-prediction uncertainty (σ)."""
        return self._predict_sigma(X_features)

    def _predict_sigma(self, X_features):
        raw = self.uncertainty_model.predict(X_features)
        # σ ≈ predicted |error| scaled by sqrt(π/2) to convert MAE to std dev
        sigma = raw * np.sqrt(np.pi / 2)
        return np.maximum(sigma, self.min_sigma)

    def _raw_prob_over(self, prediction, sigma, line):
        """Raw P(over line) using normal assumption."""
        z = (line - prediction) / sigma
        return float(1.0 - norm.cdf(z))

    def prob_over(self, prediction, sigma, line, calibrate=True):
        """
        Calibrated P(over line).

        Args:
            prediction: predicted yards
            sigma: per-prediction uncertainty
            line: betting line
            calibrate: whether to apply isotonic correction
        """
        raw_p = self._raw_prob_over(prediction, sigma, line)
        if calibrate and self.isotonic is not None:
            return float(self.isotonic.predict([raw_p])[0])
        return raw_p

    def prob_over_batch(self, predictions, sigmas, lines, calibrate=True):
        """Batch version of prob_over."""
        z = (lines - predictions) / sigmas
        raw_probs = 1.0 - norm.cdf(z)
        if calibrate and self.isotonic is not None:
            return self.isotonic.predict(raw_probs)
        return raw_probs

    def find_line_for_prob(self, prediction, sigma, target_prob, calibrate=True):
        """Find the line L such that P(over L) ≈ target_prob."""
        # Binary search
        lo, hi = prediction - 5 * sigma, prediction + 5 * sigma
        for _ in range(50):
            mid = (lo + hi) / 2
            p = self.prob_over(prediction, sigma, mid, calibrate=calibrate)
            if p > target_prob:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "uncertainty_model": self.uncertainty_model,
                "isotonic": self.isotonic,
                "min_sigma": self.min_sigma,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.uncertainty_model = data["uncertainty_model"]
        self.isotonic = data["isotonic"]
        self.min_sigma = data["min_sigma"]
        return self
