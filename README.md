# krebs — Interpretable Cancer Subtype Classification with Vector Symbolic Architectures

> **Status:** Pre-alpha research note, in active development. Not for clinical use.

A research exploration of whether **Vector Symbolic Architectures (VSA)** — also called Hyperdimensional Computing — can produce competitive **and intrinsically interpretable** classifiers for cancer molecular subtypes from gene expression data.

## Motivation

Black-box models (deep nets, gradient-boosted trees) dominate molecular oncology benchmarks but face real adoption resistance in clinical settings: oncologists need to understand *why* a model assigns a given subtype, not just trust an accuracy number. Post-hoc explainers (SHAP, LIME, attention maps) approximate these explanations after the fact.

VSA takes a different route: every prediction is a closed-form algebraic expression over high-dimensional vectors. The classifier itself *is* the explanation. This project asks: **how does that trade off against XGBoost on a real oncology benchmark?**

## Approach

- **Dataset:** TCGA-BRCA RNA-seq expression + PAM50 molecular subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like). Publicly available via UCSC Xena.
- **Baseline:** XGBoost on the top-N most variable genes, with SHAP feature importances.
- **VSA classifier:** level-encoding of expression values → patient hypervectors via binding/bundling → per-class prototype hypervectors → cosine classification.
- **Interpretability comparison:** which features each model relies on, and whether either independently recovers the published 50-gene PAM50 signature.

## Status

| Week | Goal | State |
|---|---|---|
| 1 | Data loader + XGBoost baseline | not started |
| 2 | VSA classifier | not started |
| 3 | Interpretability analysis + PAM50 recovery | not started |
| 4 | Writeup + reproducible run | not started |

## Honest framing

I'm an ML engineer with a software-engineering background, no formal training in biology or medicine. This project is a **methodology exploration**, not a clinical claim. Where domain expertise is required for biological validity, the writeup will say so explicitly.

The VSA core is vendored from my earlier project **PRISM** ([`Archive/prism`](../Archive/prism), MIT-licensed) — a from-scratch neural-free cognitive architecture I built to explore symbolic reasoning over high-dimensional vectors. This repository extends that work to numeric feature classification.

## Repo layout

```
src/krebs/
├── vsa/             # Vendored VSA primitives from PRISM
├── encoding/        # Level/feature encoders for continuous data
├── classify/        # Prototype-based VSA classifier
└── data/            # TCGA-BRCA loader

scripts/             # Download + experiment runners
tests/               # Unit + smoke tests
data/                # Local data cache (gitignored)
results/             # Metrics and figures
```

## Reproducibility

```bash
uv venv && uv pip install -e ".[dev]"
python scripts/download_data.py    # ~200MB into data/raw/
python scripts/run_experiment.py   # baseline + VSA, writes to results/
pytest                             # smoke tests
```

## References

- Plate, T. A. (1995). *Holographic Reduced Representations.* IEEE TNN 6(3).
- Kanerva, P. (2009). *Hyperdimensional Computing.* Cognitive Computation 1(2).
- Rahimi, A. et al. (2019). *Efficient biosignal processing using hyperdimensional computing.* IEEE Proc.
- Parker, J. S. et al. (2009). *Supervised risk predictor of breast cancer based on intrinsic subtypes (PAM50).* J Clin Oncol 27(8).
- TCGA Network (2012). *Comprehensive molecular portraits of human breast tumours.* Nature 490.

## License

MIT — see [LICENSE](LICENSE).
