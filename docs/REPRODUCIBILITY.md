# Reproducibility Guide

This document records the data provenance, deterministic settings, generated
artifacts, and verification steps for the results reported in the README.

## Environment

- Python 3.10 or newer
- Dependencies locked in `uv.lock`
- `uv` for environment creation and command execution
- OpenMP runtime for XGBoost on macOS (`brew install libomp`)

Create the exact development environment:

```bash
uv sync --locked --extra dev
```

## Data provenance

`scripts/download_data.py` downloads two public files from the UCSC Xena TCGA
hub into the ignored `data/raw/` directory:

| Local file | Source | SHA-256 |
|---|---|---|
| `expression.tsv.gz` | `TCGA.BRCA.sampleMap/HiSeqV2.gz` | `263bf67245cc4b9062676583c0ff0306f08471a26aafd5504037e1da22133746` |
| `clinical.tsv` | `TCGA.BRCA.sampleMap/BRCA_clinicalMatrix` | `39eb3be0fb86e6a577bd2cc01502a7fa5a271e1e1cba294e9dc644ad99580d7f` |

The downloader writes to a temporary `.part` file, verifies the complete SHA-256
digest, and only then moves the file into place. Existing files are also verified.

```bash
uv run python scripts/download_data.py
```

Raw data is intentionally excluded from Git. The tracked `results/` directory
contains aggregate metrics and model-level score arrays, not raw expression rows,
clinical records, sample identifiers, or patient identifiers.

## Verification sequence

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/run_experiment.py
uv run python scripts/run_federated.py
```

The centralized experiment writes:

- `results/experiment.json`
- `results/selected_genes.npy`
- `results/vsa_class_gene_scores.npy`
- `results/xgb_class_gene_scores.npy`

The federated experiment writes `results/federated.json`.

## Determinism and timing

The train/validation/test split, VSA role vectors, level hypervectors, and model
initialization use seed `42`. The exact federated VSA equality check should report
a maximum prototype drift of `0.0` for the pinned environment and data.

Wall-clock timings are not reproducibility invariants. They vary with CPU,
operating system, BLAS/OpenMP implementation, Python version, and dependency
versions. Compare accuracy, macro-F1, participating sites, sample counts, and VSA
prototype drift before comparing runtime.

## Interpretation boundaries

- The experiment is a simulation over public data held in one environment, not a
  deployed multi-institution protocol.
- Site-level hypervectors reduce raw-data movement but do not provide secure
  aggregation, differential privacy, authentication, or transport security.
- Results come from one seeded split and have not been clinically or peer reviewed.
- The per-site XGBoost model is a prediction-averaged ensemble, not a complete
  federated XGBoost implementation.

These boundaries should remain visible in derivative reports and publications.
