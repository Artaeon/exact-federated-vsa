# Exact Federated VSA

Interpretable cancer-subtype classification with Vector Symbolic Architectures (VSA), evaluated on real institutional partitions of TCGA-BRCA.

[![CI](https://github.com/Artaeon/exact-federated-vsa/actions/workflows/ci.yml/badge.svg)](https://github.com/Artaeon/exact-federated-vsa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Pre-alpha research software. This repository is not a medical device and must not be used for diagnosis, treatment, or other clinical decisions.

**Documentation:** [Reproducibility](docs/REPRODUCIBILITY.md) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

## Headline finding

On TCGA-BRCA PAM50 molecular-subtype classification, partitioned along the **real Tissue Source Site (TSS) code** of each tumor — i.e. the 19 institutions that actually contributed samples to TCGA — a VSA classifier admits **exact** federated training:

| Setting | Test acc | Test macro-F1 | Sites used | Training samples used |
|---|---|---|---|---|
| VSA centralized | 0.715 | 0.746 | (data pooled) | 668 / 668 |
| **VSA federated** | **0.715** | **0.746** | **19 / 19** | **668 / 668** |
| XGBoost centralized | 0.840 | 0.824 | (data pooled) | 668 / 668 |
| XGBoost per-site ensemble (prediction averaging) | 0.715 | 0.597 | 9 / 19 | 521 / 668 |

**The drift between the centralized and federated VSA models, measured as the max absolute difference across all class-prototype components, is 0.0 — bit-identical.** This is not an empirical close-match; it is the algebraic property `sum_s(sum_{class=c} H_i) = sum_{class=c} H_i` realized on real data.

Why it matters:

1. VSA federation can include **every** site, including institutions with only a handful of patients — they simply contribute fewer hypervectors. The per-site XGBoost baseline requires enough data and every class at each participating site, so it excludes 10 sites and 147 training samples here.
2. In the fair federated regime, VSA matches XGBoost accuracy and **beats it on macro-F1 by 14.9 points** (0.746 vs 0.597). XGBoost's small-site degradation falls hardest on rare classes (Her2 7%, Normal 12%).
3. The simulated VSA protocol exchanges per-class summed hypervectors rather than individual patient records. This reduces data movement, but is not by itself a production privacy guarantee.

## What this is

A direct empirical comparison of two classifier families on a question that matters to real cancer research:

> *Can multiple hospitals jointly train a tumor-subtype classifier without ever pooling patient data — and without losing accuracy?*

For Vector Symbolic Architectures (VSA, also called Hyperdimensional Computing), the model aggregation is exact by mathematical construction. The federated and centralized models are not approximately equivalent — they are the same model. The comparison baseline is a simple ensemble of per-site gradient-boosted trees, not a full implementation of a federated XGBoost protocol.

## Motivation

Black-box classifiers dominate molecular-oncology benchmarks but face two structural problems in clinical research:

1. **Trust.** Oncologists need to understand *why* a model assigned a subtype. Post-hoc explainers (SHAP, LIME) approximate the explanation after the fact; the explanation can shift with the explainer.
2. **Privacy.** Tumor data is regulated and rarely leaves the institution that collected it. Multi-site studies are valuable but logistically and legally hard.

VSA addresses both at once:

- Every prediction is a closed-form algebraic expression over high-dimensional vectors. The classifier *is* the explanation.
- Bundling (the operation that builds class prototypes) is associative, so per-site partial sums can be combined into a global classifier without revealing per-patient data.

## Method

**Data.** TCGA-BRCA RNA-seq (HiSeqV2, log2-normalized RSEM) from UCSC Xena: 956 tumor samples × 20,530 genes, stratified 70/15/15 train/val/test (seeded). Labels from `PAM50Call_RNAseq`. Heavily imbalanced (LumA 45%, Her2 7%).

**Feature selection.** Top 500 highest-variance genes on the training fold only.

**Centralized models.**
- **XGBoost baseline:** 200 trees, depth 4, lr 0.1, `multi:softprob`, with SHAP signed importances per class.
- **VSA classifier:**
    - 20 level hypervectors per gene (d=10,000, thermometer-style — adjacent levels share most bits, distant levels are nearly orthogonal).
    - Random bipolar role vector per gene; patient hypervector `H = Σ_g role_g ⊙ level(expr_g)` where `⊙` is MAP-C binding (element-wise multiply on bipolar vectors). One-shot training.
    - Class prototype `P_c = Σ_{i: y_i=c} H_i`. Predict by `argmax_c cos(H_x, P_c)`.

**Federation.** Train samples partitioned by TSS code. VSA federation: all 19 sites participate; each site emits a per-class summed hypervector; coordinator adds them. XGBoost federation: sites with ≥30 samples and all classes represented train local models; predictions averaged at inference. The same shared seed (for VSA: roles and levels; for XGBoost: tree initialization) is broadcast once before training in both schemes.

**Comparison protocol.** All four conditions evaluate on the same held-out test set. The mathematical-exactness claim is verified by computing the max absolute difference between the centralized and federated VSA prototypes.

## Results in full

### Centralized baselines

| | val acc | val macro-F1 | test acc | test macro-F1 | train time |
|---|---|---|---|---|---|
| XGBoost | 0.840 | 0.839 | 0.840 | 0.824 | 4.4 s |
| VSA | 0.833 | 0.853 | 0.715 | 0.746 | 1.8 s |

In the standard centralized setup, XGBoost outperforms VSA on raw accuracy by 12.5 points but the gap on macro-F1 shrinks to 7.8 points. VSA trained roughly 2.5× faster in this run as a one-shot prototype. Timings are hardware- and version-dependent; gradient-boosted trees remain a strong tabular baseline.

### Federated comparison (the main result)

| Setting | Test acc | Test macro-F1 | Sites used | Train samples used | Federation cost |
|---|---|---|---|---|---|
| VSA centralized | 0.715 | 0.746 | — | 668 | (baseline) |
| **VSA federated** | **0.715** | **0.746** | **all 19** | **668** | **0 (exact)** |
| XGBoost centralized | 0.840 | 0.824 | — | 668 | (baseline) |
| XGBoost per-site ensemble | 0.715 | 0.597 | 9 (≥ 30 samples, all classes) | 521 | 0.125 acc, 0.227 macro-F1 |

Site distribution (19 contributing TCGA institutions, 1–209 patients each):

```
BH 143  A8 60  E2 59  A2 53  D8 49  E9 47  AR 43  C8 37  B6 36
AO 31   AN 30  EW 21  A7 16  AC 14  GM 10  A1 10  AQ 6   GI 2   HN 1
```

### Interpretability snapshot

Per-class top genes by signed importance (XGBoost SHAP, VSA prototype unbinding):

```
xgb top-5 per class:
  LumA    SLC44A4, ERBB4, NAT1*, A2ML1, PLCH1
  LumB    KRT5*, ESR1*, KRT17*, CLEC3A, FOXJ1
  Basal   FOXA1*, BCL11A, AGR3, TFF3, TPSG1
  Her2    NPY1R, PNMT, ELOVL2, ESR1*, C2orf54
  Normal  C2orf40, FOXA1*, CAPN6, GRIA4, KRT5*
  (* = in published PAM50 signature; union recovery = 20%)

vsa top-5 per class:
  LumA    MAGEA3, GPR26, APOD, CLIC6, CACNG4
  LumB    PROL1, AGR2, PI3, CARTPT, ABCA13
  Basal   PROM1, PROL1, CST9, CARTPT, KCNJ3
  Her2    APOD, GPR26, PROL1, NBPF6, CARTPT
  Normal  MAGEA3, APOD, C7, ABCC13, PLIN4
  (union recovery = 8%)
```

XGBoost's SHAP-based importances surface canonical breast cancer markers (ESR1, KRT5, KRT17, FOXA1, NAT1 — all in the published PAM50 50-gene signature). VSA's intrinsic unbinding finds different, less canonical genes with smaller magnitudes. The intrinsic-interpretability trace is real but noisier than gradient-tree post-hoc explanation. Plausible causes and concrete next steps are listed under *Limitations* below.

## What this means for cancer research

**Federated learning introduces practical friction in multi-institutional cancer studies.** The experiment demonstrates a particularly simple aggregation property: after a shared encoding is established, the VSA prototype model can be assembled by summing site-level contributions. A real deployment would still require authentication, secure transport, governance, and privacy-preserving aggregation.

**The practical scenario this supports** is a multi-site study where:

- Each participating site holds its own RNA-seq + clinical metadata locally.
- A coordinator distributes a shared schema (role + level hypervectors) once.
- Each site computes and transmits 5 per-class summed hypervectors (~40 KB each as float32 at d=10,000, or ~200 KB total before protocol overhead).
- The coordinator sums them; the resulting classifier is bit-identical to centralized training.
- New sites can join later by adding their per-class sums to the global prototypes — no retraining.

The accuracy ceiling here (0.715 test on PAM50) is lower than gradient-boosted trees can reach centrally. Against the simple per-site ensemble used in this experiment, the VSA classifier improves macro-F1 by 14.9 points while including every institution and losing no training data. This result should not be generalized to other federated tree-learning systems without a direct comparison.

## Limitations (honest)

- **Label provenance.** `PAM50Call_RNAseq` is computed from the same expression matrix used as features. This is partial circularity — both models learn what amounts to a re-derivation of the PAM50 algorithm rather than predicting subtype from independent biological signal. A cleaner setup would use IHC-derived ER/PR/HER2 status as labels, or the older `PAM50_mRNA_nature2012` column (fewer samples but published upstream of this dataset).
- **Single split.** All numbers are from one stratified split (seed=42). A bootstrap over splits would be the next step; the val→test gap for VSA (0.833 → 0.715) suggests non-trivial fold variance that we have not yet quantified.
- **Interpretability is weaker than SHAP.** VSA's intrinsic per-(class, gene) score is noisier and recovers fewer canonical genes than XGBoost+SHAP. Concrete fixes: ensemble VSA over multiple random role-vector seeds and aggregate the sign and median magnitude per gene; or replace variance-based feature selection with a supervised ANOVA-F filter on the training fold.
- **Centralized accuracy.** VSA trails XGBoost by ~12 points centrally. The federated comparison is where VSA wins; on a single-institution single-machine setup with no privacy constraint, XGBoost remains the better classifier. The argument is structural, not "always better."
- **Federation simulation, not deployment.** We simulate federation on data we hold centrally. A real deployment needs encrypted aggregation (so the coordinator can't reverse-engineer a site's contribution from the prototype sum), and possibly differential privacy on the per-site hypervectors. Those are well-studied additions to bundle-sum protocols.
- **Domain review.** The implementation and results have not undergone formal biological, clinical, or peer review. Clinical-sounding interpretations remain conditional on independent domain-expert validation; canonical gene lists are cited from Parker et al. 2009.

## What I would do next

1. **Bootstrap the splits.** Run 50 stratified splits; report mean ± std for all four conditions. The VSA val→test gap suggests this matters.
2. **Multi-seed VSA ensembling.** Average per-(class, gene) scores across 10–50 independent role-vector seeds; expected to lift PAM50-recovery substantially.
3. **External validation.** Repeat on METABRIC, which has independent PAM50 calls — a real cross-cohort test.
4. **Add abstention.** Hold out one PAM50 class entirely; measure each model's "I don't know" behavior. VSA's cosine score has a natural threshold; clinically relevant.
5. **Multi-modal hypervectors.** Bind gene expression and clinical (stage, age, ER/PR/HER2 IHC) into one patient hypervector. The TCGA clinical table we already loaded supports this; pathology images are a larger lift.

## Source-project attribution

The VSA primitives (`bind`, `unbind`, `bundle`, `similarity`) are vendored under `src/krebs/vsa/` from the earlier MIT-licensed project [PRISM](https://github.com/Artaeon/prism). PRISM uses Plate's HRR (FFT-based circular convolution) for symbolic role-filler reasoning. This project adds MAP-C binding (element-wise multiplication on bipolar vectors) for high-throughput numeric encoding and applies the framework to TCGA-BRCA.

## Repo layout

```
src/krebs/
├── vsa/             # Vendored VSA primitives from PRISM (HRR + MAP-C)
├── encoding/        # Level + patient encoders
├── classify/        # VSA prototype classifier + XGBoost baseline
├── data/            # TCGA-BRCA loader (UCSC Xena format)
├── federated.py     # Site partitioning + federated VSA + federated XGBoost
├── features.py      # Variance-based gene filter
└── interpret.py     # Per-(class, gene) scores + PAM50 recovery

scripts/
├── download_data.py     # Pulls and verifies TCGA-BRCA from UCSC Xena (~62 MB)
├── run_experiment.py    # Centralized comparison + interpretability
└── run_federated.py     # Federated comparison

tests/                # 14 unit + smoke tests
data/                 # Local data cache (gitignored)
results/              # Metrics, signed scores, federated.json, experiment.json
```

## Reproducing

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py       # ~62 MB into ignored data/raw/
uv run pytest                                # 14 tests
uv run python scripts/run_experiment.py      # writes results/experiment.json
uv run python scripts/run_federated.py       # writes results/federated.json
```

On macOS, XGBoost also requires the OpenMP runtime:

```bash
brew install libomp
```

The downloader validates both public UCSC Xena files against pinned SHA-256 checksums and never commits the raw data. Reported timings are representative only and depend on the hardware and software environment.

For the complete data-provenance record, expected artifacts, determinism notes,
and verification checklist, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Contributing and security

Research corrections, reproducibility improvements, additional baselines, and
well-scoped implementation changes are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request. Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md); do not include controlled-access or patient-level data
in issues, pull requests, or test fixtures.

## References

- Plate, T. A. (1995). *Holographic Reduced Representations.* IEEE TNN 6(3).
- Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation 1(2).
- Rahimi, A. et al. (2019). *Efficient biosignal processing using hyperdimensional computing.* Proc IEEE.
- Imani, M. et al. (2018). *HDNA: A hyperdimensional computing-based DNA pattern matching.* DAC.
- Parker, J. S. et al. (2009). *Supervised risk predictor of breast cancer based on intrinsic subtypes (PAM50).* J Clin Oncol 27(8).
- TCGA Network (2012). *Comprehensive molecular portraits of human breast tumours.* Nature 490.
- Cheng, K. et al. (2021). *SecureBoost: A lossless federated learning framework.* IEEE Intelligent Systems 36(6).

## License

MIT — see [LICENSE](LICENSE).
