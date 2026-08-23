"""Level encoding for continuous values into hypervectors.

For each scalar input, returns a bipolar hypervector chosen from N
pre-generated **level vectors** L_0, ..., L_{N-1}. Adjacent levels share
most bits; distant levels are nearly orthogonal — so the cosine similarity
between two level vectors is monotonic in the distance between the values
they represent (this is what gives the classifier its smoothness).

Construction (Rachkovskij-style linear / "thermometer" encoding):
    L_0 is random bipolar (+1/-1).
    L_{k+1} is L_k with d/(N-1) randomly chosen bits flipped.
    Result: similarity decays roughly linearly from L_0 to L_{N-1}.

Bin edges are computed from the TRAINING distribution per feature
(global quantiles across all features by default — switchable to
per-feature quantiles, but global keeps the encoder small and stable).
"""

from __future__ import annotations

import numpy as np

from krebs.vsa import VSAConfig
from krebs.vsa.ops import HVector


class LevelEncoder:
    def __init__(
        self,
        n_levels: int = 20,
        config: VSAConfig | None = None,
    ) -> None:
        self.n_levels = n_levels
        self.config = config or VSAConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.levels: np.ndarray = self._build_levels()
        self.bin_edges: np.ndarray | None = None  # set by .fit()

    @property
    def dim(self) -> int:
        return self.config.dimension

    def _build_levels(self) -> np.ndarray:
        """Generate N level hypervectors with linearly-decaying similarity."""
        levels = np.empty((self.n_levels, self.dim), dtype=np.float32)
        levels[0] = self._rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        flips_per_step = self.dim // (self.n_levels - 1) if self.n_levels > 1 else 0
        all_indices = np.arange(self.dim)
        for k in range(1, self.n_levels):
            levels[k] = levels[k - 1].copy()
            flip_idx = self._rng.choice(all_indices, size=flips_per_step, replace=False)
            levels[k, flip_idx] *= -1
        return levels

    def fit(self, X: np.ndarray) -> LevelEncoder:
        """Compute quantile bin edges over the GLOBAL training distribution.

        Global bins (rather than per-feature) are appropriate here because
        TCGA expression is already log2-normalized, putting all genes on a
        comparable scale.
        """
        flat = X.reshape(-1)
        # n_levels bins -> n_levels-1 interior edges
        quantiles = np.linspace(0, 1, self.n_levels + 1)[1:-1]
        self.bin_edges = np.quantile(flat, quantiles).astype(np.float32)
        return self

    def encode(self, value: float | np.ndarray) -> HVector | np.ndarray:
        """Map a value (or array of values) to its level hypervector(s).

        For a scalar -> (dim,) vector.
        For shape (n,) input -> (n, dim) output.
        """
        if self.bin_edges is None:
            raise RuntimeError("LevelEncoder not fitted — call .fit(X_train) first.")
        idx = np.searchsorted(self.bin_edges, value)
        # idx may be a scalar or array depending on input
        return self.levels[idx]
