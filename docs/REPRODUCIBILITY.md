# Reproducibility and experiments

## Stored contract

Every autonomous result stores:

- Analysis ID and immutable dataset ID/SHA-256/version.
- Typed plan, objective, task/outcome decision and configuration.
- Fixed seed 42, Python/library versions and engine version.
- Feature/profile schema, dataset timestamp and the run's stored execution timestamps.
- A configuration hash over dataset hash, plan/configuration, seed and library/engine versions.
- Evidence, hypotheses, decisions, sample scopes, model parameters/scores and artifacts.

Each registered dataset is immutable and has version 1 within its dataset identity. Reuploading creates another identity; equal uploaded bytes can share the content hash. Cached profiles are lazily upgraded to the current profiling schema without replacing the underlying Parquet dataset.

Task 1 uses engine release **3.0.1**, defined centrally in `app/version.py` and included in analytical configuration hashes. Model-package schema remains 3.0 for compatibility. Older chronological evaluations may have split equal timestamps across holdouts/folds; retain those artifacts and rerun them for corrected timestamp-atomic evaluation rather than treating their prior scores as equivalent to the patch release.

## Re-run behavior

`POST /api/analysis-runs/{id}/rerun` creates a new owned queued run from stored configuration and objective. It records `parent_run_id` before committing queued work and retains the experiment. Active runs must finish or be cancelled first. An optional `budget_seconds` override intentionally changes the configuration.

Analytical values should be equivalent for the same immutable dataset/configuration/versions. Run IDs, timestamps and wall-clock timings differ. Search budgets are wall-clock boundaries between candidates; different hardware/load can change which candidates finish, so exact model-selection equivalence additionally requires the same eligible candidate set to complete. The code never promises bitwise equivalence across scientific-library versions.

## Experiments

An additive `experiments` table stores owner, name, dataset and creation time. Run membership is stored in existing configuration JSON; no destructive run-table migration is required. Autonomous runs create an experiment when none is supplied. Explicit experiments can accept additional runs from the same owned dataset.

Comparison shows model, parameters within the run, CV mean/SD, test metrics, duration and dataset hash. `best_run_id` uses development selection score only. Different target, primary metric or split contracts are not ranked together. Descriptive investigations have no fictional predictive best score.

## Export contracts

- All autonomous runs export `analysis.json`, `reproducibility.json`, metrics/plan evidence, README and escaped HTML report in a ZIP.
- Predictive runs additionally include the trusted pipeline, portable feature transformer, schema, standalone inference script and pinned inference dependencies including SciPy.
- Standalone predictions include model version and evaluation context while preserving the original prediction list format.
- Descriptive packages contain no fake pipeline or prediction service. Model prediction/download APIs return an explicit conflict when the run did not require a model; analytical report/package endpoints remain available.

## Reproducing checks

From `backend`: `uv run python -m pytest -q` and `uv run ruff check app tests`.

From `frontend`: `npm run build`; with the container running, set `PLAYWRIGHT_BASE_URL=http://localhost:8080` and run `npm run test:e2e`.

On WSL, keep Python environments and Playwright browser/system-library caches on the Linux disk filesystem rather than a Windows-mounted folder or RAM-backed `/tmp`, for example `UV_PROJECT_ENVIRONMENT=$HOME/.cache/analytiq-venv` and `PLAYWRIGHT_BROWSERS_PATH=$HOME/.cache/analytiq-playwright`. Run heavyweight checks sequentially on smaller machines. Bounded tests include an explicit evidence-equivalence re-run and independently executed exported inference.
