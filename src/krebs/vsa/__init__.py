"""Vector Symbolic Architecture primitives.

Vendored from PRISM (https://github.com/Artaeon/prism, MIT-licensed) and
trimmed to the four operations this project actually uses: bind, unbind,
bundle, similarity. Kept dependency-free (NumPy only) so the rest of this
repo can install without spaCy or any NLP stack.
"""

from krebs.vsa.config import VSAConfig, DEFAULT_CONFIG
from krebs.vsa.ops import (
    HVector,
    VectorOps,
    bind,
    bind_map,
    bundle,
    similarity,
    unbind,
    unbind_map,
)

__all__ = [
    "VSAConfig",
    "DEFAULT_CONFIG",
    "VectorOps",
    "HVector",
    "bind",
    "bind_map",
    "unbind",
    "unbind_map",
    "bundle",
    "similarity",
]
