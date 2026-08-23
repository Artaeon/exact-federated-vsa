"""Smoke tests: confirms the vendored VSA primitives behave correctly.

If these pass, bind/unbind round-trip and similarity gives near-zero for
random pairs and ~1 for self. Anything downstream (encoders, classifier)
builds on these.
"""

from krebs.vsa import DEFAULT_CONFIG, VectorOps, VSAConfig


def test_random_vectors_are_near_orthogonal() -> None:
    ops = VectorOps(VSAConfig(dimension=10_000, seed=0))
    a = ops.random_vector()
    b = ops.random_vector()
    assert abs(ops.similarity(a, b)) < 0.05


def test_self_similarity_is_one() -> None:
    ops = VectorOps(VSAConfig(dimension=10_000, seed=0))
    a = ops.random_vector()
    assert ops.similarity(a, a) > 0.999


def test_bind_unbind_roundtrip() -> None:
    # FFT-based unbind returns a noisy real vector, not bipolar; in HDC the
    # round-trip is approximate and feeds a clean-up memory. ~0.7 vs ~0 for
    # a random comparison is the expected signal at d=10k.
    ops = VectorOps(VSAConfig(dimension=10_000, seed=0))
    a = ops.random_vector()
    b = ops.random_vector()
    bound = ops.bind(a, b)
    recovered = ops.unbind(bound, b)
    other = ops.random_vector()
    sim_correct = ops.similarity(recovered, a)
    sim_other = ops.similarity(recovered, other)
    assert sim_correct > 0.5
    assert sim_correct > sim_other + 0.3


def test_bundle_preserves_similarity_to_members() -> None:
    ops = VectorOps(VSAConfig(dimension=10_000, seed=0))
    a = ops.random_vector()
    b = ops.random_vector()
    c = ops.random_vector()
    bundled = ops.bundle([a, b, c])
    assert ops.similarity(bundled, a) > 0.3
    assert ops.similarity(bundled, b) > 0.3
    assert ops.similarity(bundled, c) > 0.3


def test_default_config_has_expected_dim() -> None:
    assert DEFAULT_CONFIG.dimension == 10_000
