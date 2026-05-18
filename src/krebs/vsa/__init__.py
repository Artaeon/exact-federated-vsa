"""Vector Symbolic Architecture primitives.

Vendored from PRISM (https://github.com/Artaeon/prism, MIT-licensed) and
trimmed to the four operations this project actually uses: bind, unbind,
bundle, similarity. Kept dependency-free (NumPy only) so the rest of this
repo can install without spaCy or any NLP stack.
"""

from krebs.vsa.config import VSAConfig, DEFAULT_CONFIG
from krebs.vsa.ops import VectorOps, HVector, bind, unbind, bundle, similarity

__all__ = [
    "VSAConfig",
    "DEFAULT_CONFIG",
    "VectorOps",
    "HVector",
    "bind",
    "unbind",
    "bundle",
    "similarity",
]
