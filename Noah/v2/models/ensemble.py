"""Weighted ensemble of NN + GBM predictions."""

import numpy as np
from scipy.optimize import minimize_scalar

from v2.config import ENSEMBLE_WEIGHTS


class EnsembleModel:
    def __init__(self, nn_weight=None, gbm_weight=None):
        self.nn_weight = nn_weight or ENSEMBLE_WEIGHTS["nn"]
        self.gbm_weight = gbm_weight or ENSEMBLE_WEIGHTS["gbm"]

    def predict(self, nn_preds, gbm_preds):
        return self.nn_weight * nn_preds + self.gbm_weight * gbm_preds

    @staticmethod
    def optimize_weights(nn_preds, gbm_preds, targets):
        """Find the NN weight that minimizes MAE on a holdout set."""
        def mae_for_weight(w):
            combined = w * nn_preds + (1 - w) * gbm_preds
            return np.mean(np.abs(combined - targets))

        result = minimize_scalar(mae_for_weight, bounds=(0.0, 1.0), method="bounded")
        best_w = result.x
        return EnsembleModel(nn_weight=best_w, gbm_weight=1.0 - best_w)
