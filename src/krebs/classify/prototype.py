"""Prototype-based VSA classifier.

One-shot training (no gradients):
    For each class c:
        prototype_c = sum of patient hypervectors with class c (training set)

Prediction:
    argmax_c  cosine_similarity(patient_hv, prototype_c)

Properties:
    - Training cost: O(n_train * dim)
    - Prediction cost: O(n_classes * dim) per patient
    - Every prediction is a closed-form similarity over hypervectors;
      no learned weights, no hidden state
    - Sklearn-shaped (fit / predict / predict_proba) for parity with the
      XGBoost baseline
"""

from __future__ import annotations

import numpy as np

from krebs.encoding.patient_encoder import PatientEncoder


class VSAClassifier:
    def __init__(self, patient_encoder: PatientEncoder, n_classes: int) -> None:
        self.encoder = patient_encoder
        self.n_classes = n_classes
        self.prototypes: np.ndarray | None = None  # (n_classes, dim)

    @property
    def dim(self) -> int:
        return self.encoder.dim

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VSAClassifier":
        H = self.encoder.encode_batch(X)  # (n_samples, dim)
        protos = np.zeros((self.n_classes, self.dim), dtype=np.float32)
        for c in range(self.n_classes):
            mask = y == c
            if mask.any():
                protos[c] = H[mask].sum(axis=0)
        self.prototypes = protos
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Cosine similarity of each sample against each class prototype."""
        if self.prototypes is None:
            raise RuntimeError("VSAClassifier not fitted")
        H = self.encoder.encode_batch(X)                                  # (n, dim)
        H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        P_norm = self.prototypes / (
            np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-12
        )
        return H_norm @ P_norm.T                                          # (n, n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax over class similarities. Not strictly probabilistic
        (similarities aren't log-likelihoods) but a useful comparable score."""
        sims = self.decision_function(X)
        # Temperature 1.0; could be tuned on validation.
        sims = sims - sims.max(axis=1, keepdims=True)
        exp = np.exp(sims)
        return exp / exp.sum(axis=1, keepdims=True)
