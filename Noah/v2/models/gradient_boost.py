"""LightGBM wrapper with a scikit-learn-like interface."""

import os
import pickle
import numpy as np
import lightgbm as lgb

from v2.config import GBM_PARAMS


class GradientBoostModel:
    def __init__(self, params=None):
        self.params = params or GBM_PARAMS.copy()
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        callbacks = [lgb.log_evaluation(period=0)]
        if self.params.get("early_stopping_rounds") and X_val is not None:
            callbacks.append(
                lgb.early_stopping(self.params.pop("early_stopping_rounds", 50))
            )

        self.model = lgb.LGBMRegressor(**self.params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self
