"""VSA configuration."""

from dataclasses import dataclass


@dataclass
class VSAConfig:
    dimension: int = 10_000
    similarity_threshold: float = 0.2
    seed: int | None = None


DEFAULT_CONFIG = VSAConfig()
