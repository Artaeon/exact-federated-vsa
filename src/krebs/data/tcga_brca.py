"""TCGA-BRCA loader.

Placeholder for Week 1. Plan: read RNA-seq expression matrix (genes x
samples) and PAM50 subtype labels from data/raw/ (populated by
scripts/download_data.py). Return a tidy (X, y, gene_names) tuple with
matched sample IDs and a documented train/val/test split.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
