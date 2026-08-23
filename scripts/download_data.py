"""Download TCGA-BRCA expression + PAM50 subtype labels from UCSC Xena.

Public, no auth. Writes raw files into data/raw/.

Sources (UCSC Xena, TCGA Hub):
    Expression:   gene-level RSEM (Hugo symbols), log2(x+1) normalized
                  https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz
    Phenotype:    clinical + molecular subtype (PAM50_mRNA)
                  https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix

Downloads are verified against known SHA-256 digests before they are accepted.
"""

import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = {
    "expression.tsv.gz": (
        "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
        "263bf67245cc4b9062676583c0ff0306f08471a26aafd5504037e1da22133746",
    ),
    "clinical.tsv": (
        "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
        "39eb3be0fb86e6a577bd2cc01502a7fa5a271e1e1cba294e9dc644ad99580d7f",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, expected_sha256: str, dest: Path) -> None:
    if dest.exists():
        if sha256(dest) == expected_sha256:
            print(f"  verified: {dest.name}")
            return
        raise RuntimeError(f"Checksum mismatch for existing file: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(f"{dest.suffix}.part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                partial.open("wb") as target,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        target.write(chunk)
                        progress.update(len(chunk))

        actual_sha256 = sha256(partial)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    for name, (url, expected_sha256) in FILES.items():
        download(url, expected_sha256, RAW / name)


if __name__ == "__main__":
    main()
