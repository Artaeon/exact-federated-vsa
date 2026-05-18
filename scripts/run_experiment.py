"""End-to-end experiment: TCGA-BRCA PAM50 subtype classification.

Loads the data, applies variance-based gene filtering on the training set,
trains both classifiers (XGBoost baseline + VSA prototype), evaluates on
val and test, and writes a results JSON + per-class metrics to results/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from krebs.classify.baseline import XGBoostBaseline
from krebs.classify.prototype import VSAClassifier
from krebs.data import tcga_brca
from krebs.encoding.level_encoder import LevelEncoder
from krebs.encoding.patient_encoder import PatientEncoder
from krebs.features import top_variable_genes
from krebs.vsa import VSAConfig

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42
N_TOP_GENES = 500
VSA_DIM = 10_000
VSA_LEVELS = 20


def _metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, subtypes: list[str]) -> dict:
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": classification_report(
            y_true, y_pred, target_names=subtypes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("[1/5] loading TCGA-BRCA...")
    ds = tcga_brca.load()
    train_idx, val_idx, test_idx = tcga_brca.split(ds, seed=SEED)
    print(f"      samples: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"      classes: {ds.class_counts()}")

    print(f"[2/5] selecting top {N_TOP_GENES} variable genes (training fold only)...")
    gene_idx = top_variable_genes(ds.X[train_idx], n_top=N_TOP_GENES)
    Xtr, Xva, Xte = ds.X[train_idx][:, gene_idx], ds.X[val_idx][:, gene_idx], ds.X[test_idx][:, gene_idx]
    ytr, yva, yte = ds.y[train_idx], ds.y[val_idx], ds.y[test_idx]
    selected_genes = ds.gene_names[gene_idx]
    np.save(RESULTS_DIR / "selected_genes.npy", selected_genes)

    print("[3/5] training XGBoost baseline...")
    t0 = time.perf_counter()
    xgb_model = XGBoostBaseline(n_classes=len(ds.subtypes), seed=SEED).fit(Xtr, ytr)
    xgb_train_s = time.perf_counter() - t0

    print(f"[4/5] training VSA classifier (dim={VSA_DIM}, levels={VSA_LEVELS})...")
    cfg = VSAConfig(dimension=VSA_DIM, seed=SEED)
    level_enc = LevelEncoder(n_levels=VSA_LEVELS, config=cfg).fit(Xtr)
    pat_enc = PatientEncoder(n_genes=N_TOP_GENES, level_encoder=level_enc, config=cfg)
    t0 = time.perf_counter()
    vsa_model = VSAClassifier(pat_enc, n_classes=len(ds.subtypes)).fit(Xtr, ytr)
    vsa_train_s = time.perf_counter() - t0

    print("[5/5] evaluating...")
    results = {
        "config": {
            "seed": SEED,
            "n_top_genes": N_TOP_GENES,
            "vsa_dim": VSA_DIM,
            "vsa_levels": VSA_LEVELS,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "subtypes": ds.subtypes,
        },
        "timing_s": {"xgb_train": xgb_train_s, "vsa_train": vsa_train_s},
        "val": [
            _metrics("xgboost", yva, xgb_model.predict(Xva), ds.subtypes),
            _metrics("vsa", yva, vsa_model.predict(Xva), ds.subtypes),
        ],
        "test": [
            _metrics("xgboost", yte, xgb_model.predict(Xte), ds.subtypes),
            _metrics("vsa", yte, vsa_model.predict(Xte), ds.subtypes),
        ],
    }
    out_path = RESULTS_DIR / "experiment.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"      wrote {out_path}")

    print("\n=== val ===")
    for m in results["val"]:
        print(f"  {m['model']:<8}  acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")
    print("=== test ===")
    for m in results["test"]:
        print(f"  {m['model']:<8}  acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")
    print(f"\ntrain time:  xgb={xgb_train_s:.1f}s   vsa={vsa_train_s:.1f}s")


if __name__ == "__main__":
    main()
