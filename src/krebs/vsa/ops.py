"""Core VSA operations: bind, unbind, bundle, similarity.

Vendored from PRISM (https://github.com/Artaeon/prism, MIT-licensed) with
minor trimming. Uses bipolar (+1/-1) hypervectors of dimension d (default
10,000), with FFT-based circular convolution for binding and its inverse
for unbinding.

References:
    Kanerva, P. (2009). Hyperdimensional Computing.
    Plate, T. A. (1995). Holographic Reduced Representations.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from krebs.vsa.config import DEFAULT_CONFIG, VSAConfig

HVector = NDArray[np.float64]


class VectorOps:
    def __init__(self, config: VSAConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def dim(self) -> int:
        return self.config.dimension

    def random_vector(self) -> HVector:
        return self._rng.choice([-1.0, 1.0], size=self.dim).astype(np.float64)

    def zero_vector(self) -> HVector:
        return np.zeros(self.dim, dtype=np.float64)

    def bind(self, a: HVector, b: HVector) -> HVector:
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

    def unbind(self, bound: HVector, key: HVector) -> HVector:
        key_inv = np.roll(key[::-1], 1)
        return self.bind(bound, key_inv)

    def similarity(self, a: HVector, b: HVector) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def bundle(self, vectors: list[HVector]) -> HVector:
        if not vectors:
            return self.zero_vector()
        return np.sum(vectors, axis=0)

    def normalize(self, v: HVector) -> HVector:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def threshold(self, v: HVector) -> HVector:
        return np.sign(v).astype(np.float64)


def bind(a: HVector, b: HVector) -> HVector:
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def unbind(bound: HVector, key: HVector) -> HVector:
    key_inv = np.roll(key[::-1], 1)
    return bind(bound, key_inv)


def bundle(vectors: list[HVector]) -> HVector:
    return np.sum(vectors, axis=0) if vectors else np.zeros(0)


def similarity(a: HVector, b: HVector) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
