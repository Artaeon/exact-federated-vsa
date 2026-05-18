"""Patient encoder: gene expression matrix -> patient hypervectors.

For each patient with selected-gene expressions e_1, ..., e_F:

    patient_hv = sum_f  role_f  *  level(e_f)

where ``role_f`` is a random bipolar hypervector unique to gene f (the
"key" under which gene f's expression is stored), ``level(.)`` returns
the level hypervector corresponding to the binned value, and ``*`` is
MAP-C binding (element-wise multiply).

The sum (bundle) is left un-normalized; cosine similarity at classification
time handles magnitudes. Optional thresholding to bipolar is available for
storage compression but does not improve accuracy in our setup.
"""

from __future__ import annotations

import numpy as np

from krebs.encoding.level_encoder import LevelEncoder
from krebs.vsa import VSAConfig


class PatientEncoder:
    def __init__(
        self,
        n_genes: int,
        level_encoder: LevelEncoder,
        config: VSAConfig | None = None,
    ) -> None:
        self.n_genes = n_genes
        self.level_encoder = level_encoder
        self.config = config or level_encoder.config
        self._rng = np.random.default_rng(
            None if self.config.seed is None else self.config.seed + 1
        )
        self.roles: np.ndarray = self._rng.choice(
            [-1.0, 1.0], size=(n_genes, self.dim)
        ).astype(np.float32)

    @property
    def dim(self) -> int:
        return self.config.dimension

    def encode_one(self, x: np.ndarray) -> np.ndarray:
        """Encode a single patient: (n_genes,) -> (dim,)."""
        if x.shape[0] != self.n_genes:
            raise ValueError(f"expected {self.n_genes} features, got {x.shape[0]}")
        level_vecs = self.level_encoder.encode(x)            # (n_genes, dim)
        bound = self.roles * level_vecs                       # MAP-C bind
        return bound.sum(axis=0)                              # bundle (sum)

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode many patients: (n_samples, n_genes) -> (n_samples, dim).

        Per-sample loop to keep peak memory bounded by ~n_genes * dim floats.
        A fully-vectorized version materializes (n_samples, n_genes, dim),
        which at typical sizes (1000 x 500 x 10000) is ~19 GB and OOMs.
        """
        if X.shape[1] != self.n_genes:
            raise ValueError(f"expected {self.n_genes} features, got {X.shape[1]}")
        H = np.empty((X.shape[0], self.dim), dtype=np.float32)
        for i in range(X.shape[0]):
            H[i] = self.encode_one(X[i])
        return H
