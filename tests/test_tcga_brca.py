"""Loader smoke test against the real downloaded TCGA-BRCA data.

Skips automatically if data hasn't been downloaded yet (so CI without a
data cache won't fail). Run `python scripts/download_data.py` first.
"""
import numpy as np
import pytest

from krebs.data import tcga_brca


pytestmark = pytest.mark.skipif(
    not tcga_brca.EXPRESSION_FILE.exists() or not tcga_brca.CLINICAL_FILE.exists(),
    reason="raw TCGA-BRCA data not downloaded",
)


def test_load_shapes_and_alignment() -> None:
    ds = tcga_brca.load()
    assert ds.X.ndim == 2
    assert ds.X.shape[0] == ds.y.shape[0] == ds.sample_ids.shape[0]
    assert ds.X.shape[1] == ds.gene_names.shape[0]
    assert ds.n_samples > 800  # ~956 labeled samples expected
    assert ds.n_genes > 15_000


def test_labels_are_valid_indices() -> None:
    ds = tcga_brca.load()
    assert ds.y.min() >= 0
    assert ds.y.max() < len(ds.subtypes)
    # All five PAM50 classes should be present.
    assert len(np.unique(ds.y)) == 5


def test_class_counts_roughly_match_published() -> None:
    """Sanity check: class distribution is in the documented ballpark."""
    ds = tcga_brca.load()
    counts = ds.class_counts()
    assert counts["LumA"] > counts["Her2"]
    assert sum(counts.values()) == ds.n_samples


def test_stratified_split_preserves_class_ratio() -> None:
    ds = tcga_brca.load()
    train, val, test = tcga_brca.split(ds, seed=42)
    assert len(train) + len(val) + len(test) == ds.n_samples
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    # All five classes appear in every fold.
    assert len(np.unique(ds.y[train])) == 5
    assert len(np.unique(ds.y[val])) == 5
    assert len(np.unique(ds.y[test])) == 5
