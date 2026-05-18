"""Federated VSA experiment on TCGA-BRCA, partitioned by real TSS sites.

Compares four conditions on a shared held-out test set:
    1. Centralized VSA      — pool all training data, train one prototype set
    2. Federated VSA        — per-site prototypes, summed
    3. Centralized XGBoost  — pool all training data
    4. Federated XGBoost    — per-site models, prediction averaging

Writes results/federated.json. The headline number is the drift between
centralized and federated VSA prototypes (should be ~0 — exact equality
modulo float32 summation order).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from krebs.classify.baseline import XGBoostBaseline
from krebs.classify.prototype import VSAClassifier
from krebs.data import tcga_brca
from krebs.encoding.level_encoder import LevelEncoder
from krebs.encoding.patient_encoder import PatientEncoder
from krebs.features import top_variable_genes
from krebs.federated import (
    FederatedXGBoost,
    partition_by_site,
    train_federated_vsa,
)
from krebs.vsa import VSAConfig

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42
N_TOP_GENES = 500
VSA_DIM = 10_000
VSA_LEVELS = 20
# VSA federation can absorb arbitrarily small sites — even one patient
# contributes one hypervector. XGBoost federation needs enough per-site
# data + every class represented to train a multi-class booster, so we
# drop sites under MIN_PER_SITE_XGB. That asymmetry is the point.
MIN_PER_SITE_XGB = 30


def _metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("[1/6] loading TCGA-BRCA + split...")
    ds = tcga_brca.load()
    train_idx, val_idx, test_idx = tcga_brca.split(ds, seed=SEED)

    print(f"[2/6] feature selection (top {N_TOP_GENES} variable genes on train fold)...")
    gene_idx = top_variable_genes(ds.X[train_idx], n_top=N_TOP_GENES)
    X = ds.X[:, gene_idx]            # we slice by full-row index below
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = ds.y[train_idx], ds.y[test_idx]

    print("[3/6] partitioning train fold by TSS site...")
    # Re-key from global to train-fold-relative indices.
    g_to_local = {int(g): i for i, g in enumerate(train_idx)}

    # VSA federation: ALL sites participate (even single-patient sites).
    all_sites_global = partition_by_site(train_idx, ds.sample_ids, min_per_site=1)
    sites_vsa = {
        s: np.asarray([g_to_local[int(g)] for g in idxs])
        for s, idxs in all_sites_global.items()
    }

    # XGBoost federation: sites with enough data AND all classes.
    big_sites_global = partition_by_site(
        train_idx, ds.sample_ids, min_per_site=MIN_PER_SITE_XGB
    )
    sites_xgb = {
        s: np.asarray([g_to_local[int(g)] for g in idxs])
        for s, idxs in big_sites_global.items()
    }
    print(f"      VSA federation:   {len(sites_vsa)} sites covering "
          f"{sum(len(i) for i in sites_vsa.values())} / {len(train_idx)} training samples")
    print(f"      XGBoost federation: {len(sites_xgb)} sites covering "
          f"{sum(len(i) for i in sites_xgb.values())} / {len(train_idx)} (sites >= {MIN_PER_SITE_XGB})")
    site_counts_all = {s: int(len(i)) for s, i in sites_vsa.items()}
    for site, n in sorted(site_counts_all.items(), key=lambda kv: -kv[1]):
        marker = " (xgb)" if site in sites_xgb else ""
        print(f"        {site}  n={n}{marker}")

    print("[4/6] centralized + federated VSA...")
    cfg = VSAConfig(dimension=VSA_DIM, seed=SEED)
    level_enc = LevelEncoder(n_levels=VSA_LEVELS, config=cfg).fit(Xtr)
    pat_enc = PatientEncoder(n_genes=N_TOP_GENES, level_encoder=level_enc, config=cfg)

    t0 = time.perf_counter()
    central_vsa = VSAClassifier(pat_enc, n_classes=len(ds.subtypes)).fit(Xtr, ytr)
    t_central_vsa = time.perf_counter() - t0

    t0 = time.perf_counter()
    fed_result = train_federated_vsa(
        encoder=pat_enc,
        n_classes=len(ds.subtypes),
        X=Xtr,
        y=ytr,
        site_indices=sites_vsa,
        central_prototypes=central_vsa.prototypes,
    )
    t_fed_vsa = time.perf_counter() - t0

    fed_vsa = VSAClassifier(pat_enc, n_classes=len(ds.subtypes))
    fed_vsa.prototypes = fed_result.prototypes

    print(f"      drift (federated VSA vs centralized VSA, max abs): "
          f"{fed_result.drift_from_centralized:.6g}")

    print("[5/6] centralized + federated XGBoost...")
    t0 = time.perf_counter()
    central_xgb = XGBoostBaseline(n_classes=len(ds.subtypes), seed=SEED).fit(Xtr, ytr)
    t_central_xgb = time.perf_counter() - t0

    t0 = time.perf_counter()
    fed_xgb = FederatedXGBoost(n_classes=len(ds.subtypes), seed=SEED).fit(
        Xtr, ytr, sites_xgb
    )
    t_fed_xgb = time.perf_counter() - t0
    print(f"      federated XGBoost participating sites: {fed_xgb.n_participating_sites}")

    print("[6/6] evaluating on shared test fold...")
    rows = [
        _metrics("vsa_centralized", yte, central_vsa.predict(Xte)),
        _metrics("vsa_federated", yte, fed_vsa.predict(Xte)),
        _metrics("xgb_centralized", yte, central_xgb.predict(Xte)),
        _metrics("xgb_federated", yte, fed_xgb.predict(Xte)),
    ]
    results = {
        "config": {
            "seed": SEED,
            "n_top_genes": N_TOP_GENES,
            "vsa_dim": VSA_DIM,
            "vsa_levels": VSA_LEVELS,
            "min_per_site_xgb": MIN_PER_SITE_XGB,
            "n_sites_vsa": len(sites_vsa),
            "n_sites_xgb": len(sites_xgb),
            "site_counts_vsa": site_counts_all,
            "site_counts_xgb": {s: int(len(i)) for s, i in sites_xgb.items()},
            "n_train_total": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "subtypes": ds.subtypes,
        },
        "vsa_drift_max_abs": fed_result.drift_from_centralized,
        "timing_s": {
            "vsa_centralized": t_central_vsa,
            "vsa_federated": t_fed_vsa,
            "xgb_centralized": t_central_xgb,
            "xgb_federated": t_fed_xgb,
        },
        "test": rows,
    }
    out = RESULTS_DIR / "federated.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"      wrote {out}\n")

    print("=== test (shared test fold) ===")
    for r in rows:
        print(f"  {r['model']:<18}  acc={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}")
    print(f"\nVSA federated vs centralized drift (max abs): {fed_result.drift_from_centralized:.3e}")
    print(f"  -> {'EXACT' if fed_result.drift_from_centralized < 1e-3 else 'DRIFT'} match")


if __name__ == "__main__":
    main()
