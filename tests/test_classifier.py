"""Smoke tests for the VSA classification pipeline on synthetic data.

Verifies the full stack (level encoder + patient encoder + prototype
classifier) can separate two well-defined synthetic classes. If accuracy
is near random here, something is structurally wrong before we worry
about TCGA performance.
"""
import numpy as np

from krebs.classify.prototype import VSAClassifier
from krebs.encoding.level_encoder import LevelEncoder
from krebs.encoding.patient_encoder import PatientEncoder
from krebs.vsa import VSAConfig


def _make_two_class_data(n_per_class: int = 80, n_features: int = 50, seed: int = 7):
    rng = np.random.default_rng(seed)
    # Class 0: low expression in features [0..10], normal elsewhere
    # Class 1: high expression in features [0..10], normal elsewhere
    base = rng.normal(loc=6.0, scale=1.0, size=(2 * n_per_class, n_features)).astype(np.float32)
    base[:n_per_class, :10] -= 3.0
    base[n_per_class:, :10] += 3.0
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    perm = rng.permutation(2 * n_per_class)
    return base[perm], y[perm]


def test_vsa_classifier_separates_synthetic_classes() -> None:
    X, y = _make_two_class_data()
    cfg = VSAConfig(dimension=2_000, seed=42)
    enc = LevelEncoder(n_levels=10, config=cfg).fit(X)
    pat = PatientEncoder(n_genes=X.shape[1], level_encoder=enc, config=cfg)
    clf = VSAClassifier(pat, n_classes=2).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.9, f"expected >90% accuracy on easy synthetic data, got {acc:.2f}"


def test_predict_proba_shapes_and_sums() -> None:
    X, y = _make_two_class_data(n_per_class=20, n_features=20)
    cfg = VSAConfig(dimension=1_000, seed=42)
    enc = LevelEncoder(n_levels=8, config=cfg).fit(X)
    pat = PatientEncoder(n_genes=X.shape[1], level_encoder=enc, config=cfg)
    clf = VSAClassifier(pat, n_classes=2).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
