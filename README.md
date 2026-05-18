# krebs — Interpretable Cancer Subtype Classification with Vector Symbolic Architectures

> A methodology-comparison study: can intrinsically-interpretable VSA classifiers compete with gradient-boosted trees on a real oncology benchmark? Pre-alpha research note, not for clinical use.

## TL;DR

On TCGA-BRCA PAM50 subtype classification with the same 500 most variable genes:

| Model | val acc | val macro-F1 | test acc | test macro-F1 | train time | interpretability |
|---|---|---|---|---|---|---|
| XGBoost + SHAP | 0.847 | 0.846 | **0.840** | **0.824** | 41 s | post-hoc |
| VSA prototype | 0.833 | **0.853** | 0.715 | 0.746 | **5 s** | intrinsic |

- **VSA matches XGBoost on validation** (within 1.5%) and **beats it on macro-F1** — VSA handles class imbalance better at the chosen operating point.
- **VSA trains ~8× faster** as a one-shot prototype with zero gradient steps.
- **VSA has a larger val→test gap** (suggesting higher variance across folds — a real weakness).
- **XGBoost's SHAP recovers 20% of the canonical PAM50 signature** in its top-50 genes (ESR1, KRT5, KRT17, FOXA1, NAT1 all surface). **VSA's prototype-unbinding recovers 8%** — the signal is there but noisier than gradient-tree SHAP.

The interesting research observation is not "VSA wins/loses" but: **intrinsic algebraic interpretability is real but currently noisier than post-hoc SHAP**, which suggests concrete follow-up work rather than abandonment.

## Motivation

Black-box models dominate molecular oncology benchmarks but face real adoption resistance in clinical settings — oncologists need to understand *why* a model assigns a given subtype. Post-hoc explainers (SHAP, LIME, attention maps) approximate that explanation after the fact, and the explanation can shift with the explainer.

Vector Symbolic Architectures (VSA) take a different route: every prediction is a closed-form algebraic expression over high-dimensional vectors. The classifier itself *is* the explanation. This project compares the two paths on **TCGA-BRCA PAM50 molecular subtype classification** — a well-studied benchmark with five classes (Luminal A/B, HER2-enriched, Basal-like, Normal-like) and a published 50-gene signature to ground-truth feature importances against.

## Approach

**Data.** TCGA-BRCA RNA-seq (HiSeqV2, log2-normalized RSEM) from UCSC Xena: 956 tumor samples × 20,530 genes, stratified 70/15/15 train/val/test. Labels from the `PAM50Call_RNAseq` column. Heavily imbalanced (LumA 45%, Her2 7%).

**Feature selection.** Top 500 highest-variance genes computed on the **training fold only** (no leakage). Same 500 genes feed both models.

**XGBoost baseline.** 200 trees, depth 4, lr 0.1, `multi:softprob`. SHAP feature importances via `TreeExplainer`, averaged within each class on the test set to get a signed per-(class, gene) score.

**VSA classifier.** Sketched in `src/krebs/`:
- **Level encoder** — 20 thermometer-style hypervectors (d=10,000) at global-quantile bin edges. Adjacent levels share most bits; distant levels are nearly orthogonal, so encoder similarity is monotonic in value distance.
- **Patient encoder** — random bipolar role vector per gene; per-patient hypervector `H = Σ_g role_g ∘ level(expr_g)` where `∘` is MAP-C binding (element-wise multiply for bipolar vectors). One-shot, O(n_genes·d).
- **Prototype classifier** — `P_c = Σ_{i: y_i=c} H_i`. Predict `argmax_c cos(H_x, P_c)`. No gradients, no iteration.

**Intrinsic interpretability.** For each class `c` and gene `g`: unbind the class prototype with `role_g` and compare the result against the high-expression and low-expression level vectors. The signed cosine difference is "what level does this class's prototype expect for this gene." Every prediction can be traced back to which gene-roles contributed.

**Why MAP-C and not HRR.** PRISM (the source VSA library, vendored under `src/krebs/vsa/`) uses Plate's HRR (FFT-based circular convolution) for symbolic role-filler reasoning. For numeric classification over hundreds of features per sample, MAP-C (element-wise multiply on bipolar vectors) is O(d) instead of O(d log d), exactly self-inverse, and standard in the HDC classification literature (Rahimi et al. 2019, Imani et al. 2018). HRR primitives remain available in `vsa/ops.py`.

## Results in detail

```
=== test ===
xgboost   acc=0.840  macro_f1=0.824
vsa       acc=0.715  macro_f1=0.746

train time:  xgb=41s   vsa=5s

=== PAM50 signature recovery (top-50 genes ∩ canonical PAM50) ===
xgb top-5 per class:
  LumA    SLC44A4, ERBB4, NAT1*, A2ML1, PLCH1
  LumB    KRT5*, ESR1*, KRT17*, CLEC3A, FOXJ1
  Basal   FOXA1*, BCL11A, AGR3, TFF3, TPSG1
  Her2    NPY1R, PNMT, ELOVL2, ESR1*, C2orf54
  Normal  C2orf40, FOXA1*, CAPN6, GRIA4, KRT5*
  (* = in published PAM50; union recovery = 20%)

vsa top-5 per class:
  LumA    MAGEA3, GPR26, APOD, CLIC6, CACNG4
  LumB    PROL1, AGR2, PI3, CARTPT, ABCA13
  Basal   PROM1, PROL1, CST9, CARTPT, KCNJ3
  Her2    APOD, GPR26, PROL1, NBPF6, CARTPT
  Normal  MAGEA3, APOD, C7, ABCC13, PLIN4
  (union recovery = 8%; magnitudes ~3× smaller than SHAP)
```

## What this means

**Honest read.** The headline finding is that **the intrinsic interpretability advertised by VSA is real but currently weaker than post-hoc SHAP** in surfacing literature-validated markers. XGBoost recovers canonical PAM50 genes (ESR1, KRT5/17, FOXA1 — all expected breast cancer markers) at 2.5× the rate. VSA's per-class signatures are smaller in magnitude and feature less-canonical genes.

This is not a failure of VSA per se. Plausible causes:

1. The label column (`PAM50Call_RNAseq`) is itself derived from this expression matrix via the PAM50 algorithm. XGBoost essentially reverse-engineers that algorithm, so SHAP finds the genes the algorithm was built on. VSA's distributed bundling doesn't have the same incentive structure.
2. A single random role-vector draw introduces noise into the prototype; ensembling across draws (or supervised re-binning) is an obvious next step.
3. With 500 features and an imbalanced 5-class problem, the per-class prototype is a sum over as few as 47 samples (Her2 in the training fold). The signal-to-noise ratio is tight.

**What I'd do next** (out of scope for this 4-week study, written up to show the research direction is alive):
- Ensemble VSA over 10–50 random seeds; aggregate the per-(class, gene) score by majority sign + median magnitude.
- Try supervised feature selection (ANOVA F-statistic on the training fold) instead of pure variance.
- Compare against external labels (`PAM50_mRNA_nature2012`, fewer samples but published independently of the expression matrix).
- Investigate the test-set degradation: bootstrap the split, report mean ± std rather than a single fold.

## Honest framing

I'm a software engineer / ML engineer in Vienna. ML is largely self-taught; I have **no formal background in biology or medicine**. This project is a methodology exploration — it uses oncology data because that's the domain I want to work in, but every clinical-sounding claim is conditional on a domain expert's review. Where biological validity matters (PAM50 gene list, label provenance, subtype definitions) I have cited the upstream literature and not paraphrased.

The VSA core is vendored from my earlier project **PRISM** ([`Archive/prism`](../Archive/prism), MIT-licensed) — a from-scratch neural-free cognitive architecture I built to explore symbolic reasoning over high-dimensional vectors. This repository extends that work from symbolic facts to numeric classification.

## Repo layout

```
src/krebs/
├── vsa/             # Vendored VSA primitives from PRISM (HRR + MAP-C)
├── encoding/        # Level + patient encoders
├── classify/        # VSA prototype classifier + XGBoost baseline
├── data/            # TCGA-BRCA loader (UCSC Xena format)
├── features.py      # Variance-based gene filter
└── interpret.py     # Per-(class, gene) scores + PAM50 recovery

scripts/             # Download + end-to-end experiment
tests/               # 11 unit + smoke tests
data/                # Local data cache (gitignored)
results/             # Metrics, signed scores, selected genes
```

## Reproducing

```bash
uv venv && uv pip install -e ".[dev]"
python scripts/download_data.py       # ~62 MB into data/raw/
pytest                                # 11/11 should pass
python scripts/run_experiment.py      # ~50s total, writes results/experiment.json
```

## References

- Plate, T. A. (1995). *Holographic Reduced Representations.* IEEE TNN 6(3).
- Kanerva, P. (2009). *Hyperdimensional Computing.* Cognitive Computation 1(2).
- Rahimi, A. et al. (2019). *Efficient biosignal processing using hyperdimensional computing.* Proc IEEE.
- Imani, M. et al. (2018). *HDNA: Hyperdimensional learning for DNA pattern matching.* DAC.
- Parker, J. S. et al. (2009). *Supervised risk predictor of breast cancer based on intrinsic subtypes (PAM50).* J Clin Oncol 27(8).
- TCGA Network (2012). *Comprehensive molecular portraits of human breast tumours.* Nature 490.

## License

MIT — see [LICENSE](LICENSE).
