"""Federated training utilities.

The VSA federated protocol exploits one structural fact: bundling is
associative, so

    sum_s ( sum_{i in site_s, class c} H_i )  ==  sum_{i in train, class c} H_i

provided every site encodes its patients with the **same** role and level
hypervectors. In a real deployment a coordinator broadcasts a shared seed
once before training begins; patient data never leaves the site, only the
per-class summed hypervectors do.

This module simulates that protocol on TCGA-BRCA by partitioning samples
along their TCGA Tissue Source Site (TSS) code — the real institution
that contributed each tumor. No data is fabricated; the only fictional
element is that we centrally evaluate against the same held-out test set
the centralized model uses, which is fair for a methodology comparison.

We also implement a prediction-ensembling federated XGBoost as a baseline:
each site trains its own model, predicted probabilities are averaged at
inference. This is the standard "cheap" federated approach for tree models
and is what most production federated tree methods reduce to absent special
infrastructure (SecureBoost et al.).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from krebs.classify.prototype import VSAClassifier
from krebs.encoding.patient_encoder import PatientEncoder


def tss_codes(sample_ids: np.ndarray) -> np.ndarray:
    """Extract the TCGA Tissue Source Site (TSS) code from each sample ID.

    TCGA sample IDs look like ``TCGA-{TSS}-{participant}-{vial}``; the TSS
    code identifies the institution that contributed the tumor.
    """
    return np.asarray([str(s).split("-")[1] for s in sample_ids])


def partition_by_site(
    indices: np.ndarray,
    sample_ids: np.ndarray,
    min_per_site: int = 30,
) -> dict[str, np.ndarray]:
    """Group sample indices by TSS code, dropping under-represented sites.

    Returns ``{tss_code: index_array}``. Indices are subset of ``indices``.
    """
    tss = tss_codes(sample_ids)
    by_site: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        by_site[tss[i]].append(int(i))
    return {
        site: np.asarray(idxs)
        for site, idxs in by_site.items()
        if len(idxs) >= min_per_site
    }


@dataclass
class FederatedVSAResult:
    prototypes: np.ndarray                # (n_classes, dim)
    per_site_prototypes: dict[str, np.ndarray]   # site -> (n_classes, dim)
    per_site_sample_counts: dict[str, int]
    drift_from_centralized: float         # max abs diff vs centralized, 0 if exact


def train_federated_vsa(
    encoder: PatientEncoder,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
    site_indices: dict[str, np.ndarray],
    central_prototypes: np.ndarray | None = None,
) -> FederatedVSAResult:
    """Simulate the federated VSA protocol.

    Each site computes its local per-class prototype sums. The coordinator
    adds them together. If ``central_prototypes`` is supplied, we also
    compute the max-abs deviation between federated and centralized
    prototypes — this verifies the exact-equality property numerically.
    """
    per_site: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    fed = np.zeros((n_classes, encoder.dim), dtype=np.float32)
    for site, idxs in site_indices.items():
        Xs = X[idxs]
        ys = y[idxs]
        Hs = encoder.encode_batch(Xs)
        local = np.zeros((n_classes, encoder.dim), dtype=np.float32)
        for c in range(n_classes):
            mask = ys == c
            if mask.any():
                local[c] = Hs[mask].sum(axis=0)
        per_site[site] = local
        counts[site] = len(idxs)
        fed += local

    drift = 0.0
    if central_prototypes is not None:
        drift = float(np.abs(fed - central_prototypes).max())
    return FederatedVSAResult(
        prototypes=fed,
        per_site_prototypes=per_site,
        per_site_sample_counts=counts,
        drift_from_centralized=drift,
    )


class FederatedXGBoost:
    """Per-site XGBoost models with prediction-probability ensembling.

    This is the standard "cheap" federation for trees: train locally, average
    P(y|x) across site models at inference. Honest weakness vs centralized:
    each site sees ~1/n_sites of the training data, so individual trees are
    weaker; averaging cannot fully recover the centralized model. More
    sophisticated tree-federation (SecureBoost et al.) requires infrastructure
    beyond a scikit-shaped wrapper.
    """

    def __init__(self, n_classes: int, seed: int = 42) -> None:
        self.n_classes = n_classes
        self.seed = seed
        self.models: dict[str, xgb.XGBClassifier] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        site_indices: dict[str, np.ndarray],
    ) -> "FederatedXGBoost":
        for site, idxs in site_indices.items():
            Xs = X[idxs]
            ys = y[idxs]
            # Each site needs to see every class at least once; if not,
            # skip — that site cannot contribute a usable multiclass model.
            if len(np.unique(ys)) < self.n_classes:
                continue
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=self.n_classes,
                n_jobs=-1,
                random_state=self.seed,
                eval_metric="mlogloss",
                tree_method="hist",
            )
            model.fit(Xs, ys)
            self.models[site] = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("FederatedXGBoost has no fitted site models")
        probs = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        for model in self.models.values():
            probs += model.predict_proba(X)
        return probs / len(self.models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def n_participating_sites(self) -> int:
        return len(self.models)
