"""Download TCGA-BRCA expression + PAM50 subtype labels from UCSC Xena.

Public, no auth. Writes raw files into data/raw/.

Sources (UCSC Xena, TCGA Hub):
    Expression:   gene-level RSEM (Hugo symbols), log2(x+1) normalized
                  https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz
    Phenotype:    clinical + molecular subtype (PAM50_mRNA)
                  https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix

Placeholder: will be filled in during Week 1. Stubbed here so the README's
reproducibility section reflects the real entry point.
"""

from pathlib import Path

import requests
from tqdm import tqdm

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

URLS = {
    "expression.tsv.gz": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "clinical.tsv": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    for name, url in URLS.items():
        download(url, RAW / name)


if __name__ == "__main__":
    main()
