"""Feature selection on gene expression.

Selects the top-N most variable genes computed on the TRAINING subset only
(no test-set leakage). Variance-based selection is the standard unsupervised
gene-filter step in transcriptomics; supervised alternatives (ANOVA F,
mutual information) are easy to swap in later if needed.
"""

from __future__ import annotations

import numpy as np


def top_variable_genes(
    X_train: np.ndarray,
    n_top: int = 500,
) -> np.ndarray:
    """Return indices of the ``n_top`` highest-variance columns of X_train.

    Indices are sorted by descending variance so downstream slicing is stable.
    """
    if n_top > X_train.shape[1]:
        n_top = X_train.shape[1]
    variances = X_train.var(axis=0)
    # argpartition is O(n) vs argsort's O(n log n); for 20k genes it matters less,
    # but the resulting indices are unordered — sort them by variance for stability.
    top_unordered = np.argpartition(variances, -n_top)[-n_top:]
    return top_unordered[np.argsort(-variances[top_unordered])]
