"""XGBoost baseline classifier.

Thin sklearn-shaped wrapper so the experiment script can train / evaluate
the baseline and the VSA classifier with identical glue code.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb


class XGBoostBaseline:
    def __init__(self, n_classes: int, seed: int = 42) -> None:
        self.n_classes = n_classes
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            n_jobs=-1,
            random_state=seed,
            eval_metric="mlogloss",
            tree_method="hist",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
