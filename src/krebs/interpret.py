"""Interpretability tools for the VSA classifier and the XGBoost baseline.

Two artifacts, both per (class, gene):

1. VSA "expected level" — unbind the class prototype with each gene's role
   vector to recover the level hypervector the prototype encodes for that
   gene; pick the level with highest cosine similarity. This is intrinsic:
   the classifier IS this lookup.

2. XGBoost SHAP signed importance — TreeExplainer values averaged across
   test samples of a given class; tells which genes pushed the model toward
   that class. Post-hoc but standard.

Both are exposed as (n_classes, n_genes) score matrices so downstream code
can rank, compare overlap, and cross-reference with literature signatures.
"""

from __future__ import annotations

import numpy as np
import shap

from krebs.classify.baseline import XGBoostBaseline
from krebs.classify.prototype import VSAClassifier

# Parker et al. 2009 PAM50 gene signature (Hugo symbols).
# Reference: "Supervised risk predictor of breast cancer based on intrinsic
# subtypes", J Clin Oncol 27(8). These 50 genes drive PAM50 calls.
PAM50_GENES = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "KRT6A", "MAPT", "MDM2", "MELK", "MIA",
    "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2",
    "ORC6", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B",
    "TYMS", "UBE2C", "UBE2T",
}


def vsa_class_gene_scores(model: VSAClassifier) -> np.ndarray:
    """Per (class, gene) score: cosine(unbound(prototype, role_g), L_max) -
    cosine(unbound(prototype, role_g), L_min). Positive means the prototype
    "expects" gene g to be HIGH for that class; negative means LOW.

    Returns: (n_classes, n_genes) signed score.
    """
    if model.prototypes is None:
        raise RuntimeError("VSAClassifier not fitted")
    roles = model.encoder.roles                         # (n_genes, dim)
    levels = model.encoder.level_encoder.levels        # (n_levels, dim)
    n_classes = model.n_classes
    n_genes = roles.shape[0]
    L_low = levels[0]
    L_high = levels[-1]
    scores = np.empty((n_classes, n_genes), dtype=np.float32)
    proto_norms = np.linalg.norm(model.prototypes, axis=1, keepdims=True) + 1e-12
    P = model.prototypes / proto_norms                  # (n_classes, dim)
    roles_unit = roles / (np.linalg.norm(roles, axis=1, keepdims=True) + 1e-12)
    L_low_unit = L_low / (np.linalg.norm(L_low) + 1e-12)
    L_high_unit = L_high / (np.linalg.norm(L_high) + 1e-12)
    for c in range(n_classes):
        unbound = P[c][None, :] * roles                # MAP-C unbind: (n_genes, dim)
        u_norm = unbound / (np.linalg.norm(unbound, axis=1, keepdims=True) + 1e-12)
        sim_high = u_norm @ L_high_unit
        sim_low = u_norm @ L_low_unit
        scores[c] = sim_high - sim_low
    # roles_unit kept for clarity; unused beyond docs
    _ = roles_unit
    return scores


def xgb_class_gene_scores(
    model: XGBoostBaseline,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Mean signed SHAP value per (class, gene) computed on (X, y).

    For multiclass XGBoost, shap returns a list of (n_samples, n_genes)
    arrays — one per class. We average each per-class array over the
    samples that BELONG to that class, giving "what drove this model to
    call samples of class c as class c?".
    """
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        per_class = shap_values
    else:
        # newer shap returns (n_samples, n_genes, n_classes)
        per_class = [shap_values[..., c] for c in range(shap_values.shape[-1])]
    n_classes = len(per_class)
    n_genes = X.shape[1]
    scores = np.zeros((n_classes, n_genes), dtype=np.float32)
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            scores[c] = per_class[c][mask].mean(axis=0)
    return scores


def top_genes_per_class(
    scores: np.ndarray,
    gene_names: np.ndarray,
    k: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """For each class, return the top-k genes by |score| with signed values."""
    out: dict[int, list[tuple[str, float]]] = {}
    for c in range(scores.shape[0]):
        order = np.argsort(-np.abs(scores[c]))[:k]
        out[c] = [(str(gene_names[i]), float(scores[c, i])) for i in order]
    return out


def pam50_recovery(
    scores: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 50,
) -> dict[str, float]:
    """Fraction of each class's top-k genes that appear in the published PAM50 set.

    A model that "knows" the right oncology should rank PAM50 genes high; a
    model that fits arbitrary high-variance noise will not.
    """
    out: dict[str, float] = {}
    for c in range(scores.shape[0]):
        order = np.argsort(-np.abs(scores[c]))[:top_k]
        top = {str(gene_names[i]) for i in order}
        out[str(c)] = len(top & PAM50_GENES) / top_k
    out["union_across_classes"] = len(
        {
            str(gene_names[i])
            for c in range(scores.shape[0])
            for i in np.argsort(-np.abs(scores[c]))[:top_k]
        }
        & PAM50_GENES
    ) / len(PAM50_GENES)
    return out
