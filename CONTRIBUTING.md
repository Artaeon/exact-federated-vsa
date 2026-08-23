# Contributing

Thank you for helping improve Exact Federated VSA. Contributions should preserve
the repository's research focus, reproducibility, and non-clinical scope.

## Before opening an issue

- Use GitHub Issues for reproducible bugs, documentation corrections, and
  concrete experiment proposals.
- Do not upload patient-level, controlled-access, or otherwise sensitive data.
- Report security concerns privately according to [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/Artaeon/exact-federated-vsa.git
cd exact-federated-vsa
uv sync --locked --extra dev
uv run python scripts/download_data.py
```

On macOS, XGBoost requires the OpenMP runtime (`brew install libomp`). Raw data
is downloaded into the ignored `data/raw/` directory and verified against the
pinned SHA-256 checksums before use.

## Quality checks

Run all checks before submitting a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

If a change affects the experiment, also run both entry points and explain any
metric changes in the pull request:

```bash
uv run python scripts/run_experiment.py
uv run python scripts/run_federated.py
```

Commit updated files from `results/` only when the change intentionally alters
the published experiment. Never commit `data/raw/`, temporary downloads, local
environments, credentials, or patient/sample-level exports.

## Pull requests

- Keep each pull request focused and describe the scientific or engineering
  rationale.
- Add tests for behavioral changes and regression fixes.
- Separate measured results from interpretation; identify inferences clearly.
- State the seed, dataset version, command, and environment for new benchmarks.
- Do not present exploratory results as clinical evidence.

By contributing, you agree that your contribution is licensed under the MIT
License.
