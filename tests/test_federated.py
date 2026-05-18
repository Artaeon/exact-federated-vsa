"""Tests for the federated VSA protocol.

Core invariant: sum_s ( sum_{i in site_s, class c} H_i ) == sum_{i in train,
class c} H_i. The federated prototypes should match the centralized
prototypes to float32 precision when sites use identical role/level vectors.
"""
import numpy as np

from krebs.classify.prototype import VSAClassifier
from krebs.encoding.level_encoder import LevelEncoder
from krebs.encoding.patient_encoder import PatientEncoder
from krebs.federated import (
    partition_by_site,
    train_federated_vsa,
    tss_codes,
)
from krebs.vsa import VSAConfig


def _toy_data(n_per_class: int = 60, n_features: int = 30, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=6.0, scale=1.0, size=(3 * n_per_class, n_features)).astype(np.float32)
    X[:n_per_class, :5] += 3
    X[n_per_class:2 * n_per_class, :5] -= 3
    y = np.concatenate([
        np.zeros(n_per_class),
        np.ones(n_per_class),
        np.full(n_per_class, 2),
    ]).astype(np.int64)
    # Fake TCGA-style IDs so we can partition by "site"
    sites = ["AA", "BB", "CC", "DD"]
    sample_ids = np.asarray(
        [f"TCGA-{sites[i % len(sites)]}-{i:04d}-01" for i in range(len(y))], dtype=str
    )
    perm = rng.permutation(len(y))
    return X[perm], y[perm], sample_ids[perm]


def test_tss_extraction_from_ids() -> None:
    ids = np.asarray(["TCGA-AA-0001-01", "TCGA-BB-0002-01"], dtype=str)
    assert list(tss_codes(ids)) == ["AA", "BB"]


def test_partition_by_site_drops_small_sites() -> None:
    X, y, ids = _toy_data()
    idxs = np.arange(len(y))
    parts = partition_by_site(idxs, ids, min_per_site=10)
    assert len(parts) == 4   # all four sites are >= 10 in our toy setup
    total = sum(len(v) for v in parts.values())
    assert total == len(y)


def test_federated_vsa_equals_centralized_exactly() -> None:
    """The core mathematical claim of the federated protocol."""
    X, y, ids = _toy_data()
    n_classes = 3
    cfg = VSAConfig(dimension=2_000, seed=42)
    enc = LevelEncoder(n_levels=10, config=cfg).fit(X)
    pat = PatientEncoder(n_genes=X.shape[1], level_encoder=enc, config=cfg)

    central = VSAClassifier(pat, n_classes=n_classes).fit(X, y)

    idxs = np.arange(len(y))
    site_indices = partition_by_site(idxs, ids, min_per_site=1)
    fed = train_federated_vsa(
        encoder=pat, n_classes=n_classes,
        X=X, y=y, site_indices=site_indices,
        central_prototypes=central.prototypes,
    )
    # Float32 summation over <=180 samples per class — drift should be
    # bounded by a few ULPs of the largest partial sum (~ a few thousand).
    assert fed.drift_from_centralized < 1e-2

    # And predictions should also match exactly on the training data.
    fed_clf = VSAClassifier(pat, n_classes=n_classes)
    fed_clf.prototypes = fed.prototypes
    assert (fed_clf.predict(X) == central.predict(X)).all()
