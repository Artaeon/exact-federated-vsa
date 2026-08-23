"""TCGA-BRCA loader.

Reads RNA-seq expression and PAM50 subtype labels from data/raw/ (populated
by scripts/download_data.py). Returns a tidy dataset with matched sample
IDs and a deterministic stratified split.

Methodological note:
    The default label column ``PAM50Call_RNAseq`` is computed *from* the
    same RNA-seq matrix used here as features (UCSC Xena applies the PAM50
    classifier on the expression data). This means the supervised task is
    closer to "recover the PAM50 algorithm" than "predict subtype from
    novel expression data" — which is fine for a method-comparison study
    (XGBoost vs. VSA on identical labels), but a real clinical study would
    need independent ground truth (immunohistochemistry, ER/PR/HER2 IHC).
    Switch to ``PAM50_mRNA_nature2012`` for the published 2012 labeling
    (fewer samples, but authored upstream of this dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "processed"

EXPRESSION_FILE = RAW_DIR / "expression.tsv.gz"
CLINICAL_FILE = RAW_DIR / "clinical.tsv"

DEFAULT_LABEL_COL = "PAM50Call_RNAseq"
SUBTYPES = ["LumA", "LumB", "Basal", "Her2", "Normal"]


@dataclass
class TCGABRCA:
    """A loaded TCGA-BRCA dataset.

    Attributes
    ----------
    X : np.ndarray, shape (n_samples, n_genes)
        log2(x+1) normalized RSEM expression values.
    y : np.ndarray, shape (n_samples,)
        Integer-encoded PAM50 subtype labels (indices into ``subtypes``).
    subtypes : list[str]
        Human-readable subtype names; ``subtypes[y[i]]`` gives sample i's label.
    gene_names : np.ndarray, shape (n_genes,)
        Hugo gene symbols, column-aligned to X.
    sample_ids : np.ndarray, shape (n_samples,)
        TCGA sample IDs (e.g. ``TCGA-A1-A0SB-01``), row-aligned to X.
    """

    X: np.ndarray
    y: np.ndarray
    subtypes: list[str]
    gene_names: np.ndarray
    sample_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y, minlength=len(self.subtypes))
        return {name: int(c) for name, c in zip(self.subtypes, counts, strict=True)}


def load(
    label_col: str = DEFAULT_LABEL_COL,
    cache: bool = True,
) -> TCGABRCA:
    """Load and align the TCGA-BRCA expression + PAM50 labels.

    Drops samples with missing or unrecognized labels. Caches the parsed
    matrices as ``data/processed/tcga_brca.npz`` so repeat loads are fast
    (the expression TSV takes ~10s to parse from gzip).
    """
    if not EXPRESSION_FILE.exists() or not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Raw data missing in {RAW_DIR}. Run `python scripts/download_data.py` first."
        )

    cache_path = CACHE_DIR / f"tcga_brca_{label_col}.npz"
    if cache and cache_path.exists():
        return _from_cache(cache_path)

    # Clinical: small TSV; parse the sampleID + label columns only.
    clinical = pd.read_csv(
        CLINICAL_FILE,
        sep="\t",
        usecols=["sampleID", label_col],
        low_memory=False,
    )
    clinical = clinical.dropna(subset=[label_col])
    clinical = clinical[clinical[label_col].isin(SUBTYPES)]

    # Expression: rows are genes, columns are samples; ~20k x ~1200.
    # Read once into memory (~200MB), then transpose to samples x genes.
    expression = pd.read_csv(EXPRESSION_FILE, sep="\t", index_col=0)

    common_samples = clinical["sampleID"].values
    keep = [s for s in common_samples if s in expression.columns]
    expression = expression[keep]
    clinical = clinical[clinical["sampleID"].isin(keep)].set_index("sampleID").loc[keep]

    label_to_idx = {name: i for i, name in enumerate(SUBTYPES)}
    y = clinical[label_col].map(label_to_idx).to_numpy(dtype=np.int64)
    X = expression.to_numpy(dtype=np.float32).T  # samples x genes
    # Cast to fixed-width Unicode so the .npz cache loads without allow_pickle.
    gene_names = np.asarray(expression.index.tolist(), dtype=str)
    sample_ids = np.asarray(keep, dtype=str)

    ds = TCGABRCA(X=X, y=y, subtypes=SUBTYPES, gene_names=gene_names, sample_ids=sample_ids)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=ds.X,
            y=ds.y,
            gene_names=ds.gene_names,
            sample_ids=ds.sample_ids,
            subtypes=np.asarray(ds.subtypes, dtype=str),
        )
    return ds


def _from_cache(path: Path) -> TCGABRCA:
    npz = np.load(path, allow_pickle=False)
    return TCGABRCA(
        X=npz["X"],
        y=npz["y"],
        subtypes=list(npz["subtypes"]),
        gene_names=npz["gene_names"],
        sample_ids=npz["sample_ids"],
    )


def split(
    ds: TCGABRCA,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split. Returns three index arrays."""
    idx = np.arange(ds.n_samples)
    train_val, test = train_test_split(idx, test_size=test_size, stratify=ds.y, random_state=seed)
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=ds.y[train_val],
        random_state=seed,
    )
    return train, val, test
