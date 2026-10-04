# Verification record

## Task 1 maintenance verification — engine 3.0.1

Fresh audit baseline started at clean commit `d34c875` on October 4, 2026. Python 3.12.15, Node.js 24.21.0 and Docker Desktop's Linux engine were used.

| Check | Baseline | Fixed release |
|---|---|---|
| Backend suite, including security | 57 passed, 3 warnings, 55.95 seconds | **92 passed, 3 warnings, 39.65 seconds** |
| Ruff | Passed | Passed |
| Explicit TypeScript check | Passed | Passed |
| Vite production build | Passed, 746 modules | Passed, 746 modules; also built inside Docker |
| Chromium E2E | 6 passed in 2.4 minutes after environment recovery | **6 passed in 1.8 minutes** against the fixed image, including invalid-input recovery |
| Docker build | `analytiq:task1-baseline` built | `analytiq:task1-fixed` and `analytiq:latest` built successfully |
| Compose | Inspected both definitions | Both `config -q` validations passed |
| npm advisory scans | Full and production-only scans: no known vulnerabilities | Dependency lock unchanged |
| Python advisory scan | `pip-audit` installed environment: no known vulnerabilities | Scientific dependency versions unchanged |
| Deployment/persistence | Existing port-8080 volume inspected; no active jobs | Healthy 3.0.1 deployment; all original records/artifacts preserved |

The fixed image build used `uv sync --frozen` and `npm ci`; application version changes did not update scientific dependencies.

The running deployment uses image ID `sha256:b57426bd2bf9b094f5c8f77aa52806aeeff9fa47ec75f4c389caf212069a0a3e`, non-root user `analyst`, port 8080, `MAX_WORKERS=1` and the existing `analytiq-data:/app/data` mount. It was confirmed running/healthy. Replacement preserved exactly **30 users, 67 datasets, 25 runs, 6 experiments and 18 saved pipelines**. The full four-table row digest remained `a0602893818423b1c2b50ca6b96ed2463e227fddfa50e24ce4a11b02848f3dc5`; all registered dataset files and all completed-run reports/ZIPs/required pipelines existed. A deployment smoke check verified the oldest saved model's prediction/report/package and safe legacy failure redaction through the owned API.

The deployed health endpoint returned `{"status":"ok","version":"3.0.1","llm_enabled":false,"demo_enabled":true}`. Application: **http://localhost:8080**; documentation: **http://localhost:8080/docs**. The QA container on 8081 was stopped after verification.

Added 35 focused regression cases in `backend/tests/test_baseline_regressions.py`, covering safe structured errors/credential redaction, streaming body caps, required JWT claims, scalar inference contracts, blank/legacy names, terminal-state races, broken-pool/transient-database recovery, concurrent quotas, expired limiter capacity, experiment foreign keys, timestamp-atomic splits, feature-name collisions, explicit planner contracts, numeric class roles and rejected uploaded serialized models. Initial focused reproductions failed before their corresponding fixes. The existing desktop browser scenario additionally verifies invalid batch input receives a structured 422, displays the error and recovers to successful prediction.

The first baseline browser attempt stalled under exhausted WSL swap; a launch-only retry then lacked libraries after a WSL restart cleared `/tmp`. Moving/restoring browser dependencies to Linux disk caches resolved both environment problems, and all six baseline workflows passed against the baseline image on isolated QA port 8081. Final QA also uses the isolated test volume and one worker.

Final command from `frontend`:

```bash
LD_LIBRARY_PATH=$HOME/.cache/analytiq-browser-libs/usr/lib/x86_64-linux-gnu \
PLAYWRIGHT_BROWSERS_PATH=$HOME/.cache/analytiq-playwright \
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8081 \
NODE_OPTIONS=--max-old-space-size=512 \
npx playwright test --workers=1 --output=/tmp/omnirush/analytiq-task1-final
```

Additional scan commands: `npm audit --json`, `npm audit --omit=dev --json`, and `pip-audit --path $HOME/.cache/analytiq-venv/lib/python3.12/site-packages --progress-spinner off --format json`. Advisory results are a snapshot, not penetration-test certification. Detailed severity findings and remaining P2/P3 items are in [TASK_1_AUDIT.md](TASK_1_AUDIT.md).

## Historical functional-upgrade verification — engine 3.0.0

Verified on October 4, 2026, in WSL/Linux with Python 3.12.15 and Node.js 24.21.0. The final single-container application reports version **3.0.0** and serves the dashboard/API at **http://localhost:8080**.

## Results at a glance

| Check | Actual result |
|---|---|
| Backend tests | **57 passed, 3 warnings in 38.29 seconds** |
| Ruff application/test checks | Passed |
| TypeScript and Vite production build | Passed; 746 modules |
| Combined Chromium workflows | **6 passed in approximately 3 minutes** against the final image |
| Docker image | Final `analytiq:latest` build passed; container running and healthy |
| Compose configurations | Both single-container and multi-service configuration validation passed |
| Container replacement and restart | Persistent records, analytical state and artifacts retained |
| Large-data analytical check | 100,000 rows retained; bounded profiling/statistics/anomaly work completed in 1.257 seconds |

## Backend

The original twelve tests remain, with additional coverage in `test_planner.py`, `test_autonomous.py`, `test_analytics.py` and `test_security.py`. The complete suite contains 57 passing cases, including parametrized edge cases.

Verified behavior includes:

- CSV/Excel ingestion, advanced profiling, paginated previews, authenticated ownership and account isolation.
- Preserved classification/regression/clustering workflows, leakage-safe group/time CV, SHAP/error analysis, missing/unseen-category predictions and independent exported inference, including portable date features.
- Objective-driven planning, target ambiguity, quality-only/diagnostic/segmentation/forecast decisions, explicit ML skips and eligibility bounds.
- Completed descriptive investigations with linked evidence, hypotheses, confidence, recommendations, HTML reports and evidence packages; predictive endpoints reject descriptive runs.
- Evidence-equivalent deterministic re-runs, configuration hashes, experiment grouping, ownership and cancellation/publication guards.
- Pearson correlations and intervals, Welch/rank/multigroup tests, Fisher exact association, adjusted p-values and distinction between statistical and practical significance.
- Chronological forecast validation/test separation, missing/duplicate/future dates, temporal gaps and documented forecast skips.
- Explained anomalies with retained original rows; semantic/temporal/target-copy leakage flags without automatic feature removal.
- Severe binary imbalance with minority-positive PR-AUC, multiclass macro F1, calibration/Brier/reliability, baseline eligibility and model stability.
- JSON-safe constant/empty/mixed/free-text/non-finite/extreme-magnitude data; explicit rejection of empty ingestion and bounded large-data work.
- Malicious filenames, traversing/malformed/compressed-bomb spreadsheet archives, upload bounds, artifact path containment and authentication rate limiting.
- Optional LLM responses restricted to existing evidence IDs: accepted valid responses, rejected invented IDs/metrics/prose, no-key fallback and provider-error fallback.
- Static deep links, asset caching, missing-asset/API responses and static-directory containment.

Command used from `backend`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/diyor/.cache/analytiq-venv/bin/python -m pytest -q
/home/diyor/.cache/analytiq-venv/bin/ruff check app tests
```

The three warnings were one Starlette/httpx TestClient deprecation and two bounded MLP convergence warnings during cross-validation. They were not test failures.

## Browser

All six Chromium scenarios passed together against the final deployed image:

- Desktop: demo sign-in → exploration/data preview/computed chat → training wizard → background ML → model comparison → SHAP → prediction → evaluated noncausal scenario → ZIP export → theme change. Direct reloads, API documentation and unknown API routes were checked.
- Mobile, 390 × 844: demo sign-in → responsive navigation → dataset exploration → rendered scatter points → no page-wide horizontal overflow.
- Clustering: target-free wizard → training → group visualization → SHAP → new-record group prediction.
- Autonomous diagnostic: time-series sample → target-specific quality/leakage review → objective/plan → explicit no-ML result → evidence/trace/hypotheses/actions → report package → re-run → experiment comparison. A descriptive run also redirects safely out of the model-studio route.
- Mobile autonomous forecast: objective/plan → evaluated temporal baseline and forecast table → no page-wide horizontal overflow.
- Autonomous prediction: churn objective → justified ML decision → computed findings → preserved model studio, reliability evidence and SHAP.

Prediction checks explicitly await the completed API response and verify HTTP 200 before checking the displayed output. This accommodates first-use model loading on the resource-constrained verification machine.

Command used from `frontend`:

```bash
LD_LIBRARY_PATH=/tmp/omnirush/browser-libs/usr/lib/x86_64-linux-gnu \
PLAYWRIGHT_BROWSERS_PATH=/tmp/omnirush/playwright \
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8080 \
NODE_OPTIONS=--max-old-space-size=512 \
npx playwright test --workers=1 --output=/tmp/omnirush/analytiq-autonomous-final
```

A separate desktop diagnostic/mobile forecast visual review recorded no uncaught browser errors. Full-page layouts and individual temporal-chart screenshots were reviewed after scrolling charts into view and allowing animations to settle. These screenshots are under `/tmp/omnirush/analytiq-autonomous-final`; they are local verification artifacts, not committed application data.

## Production frontend

`NODE_OPTIONS=--max-old-space-size=768 npm run build` passed, including TypeScript compilation and Vite production bundling. Fonts remain local, analytical pages are lazy-loaded, charts/shared dependencies have separate chunks, and the final module graph contains 746 modules.

## Performance observation

A separate deterministic synthetic check used 100,000 rows, two normal numeric columns and a three-value categorical column, with seed 42 and numerical threads bounded to one:

| Operation | Observed time | Sample scope |
|---|---:|---|
| Advanced profiling | 0.997 seconds | 10,000-row distribution sample; full-row quality counts |
| Selected statistical tests | 0.098 seconds | 5,000 rows per test |
| Isolation Forest anomalies | 0.163 seconds | 10,000 rows |
| Combined analytical operations | 1.257 seconds | All 100,000 source rows retained |

Peak process RSS was 830.8 MiB, including loaded libraries and the dataframe. Timings exclude interpreter/library startup and synthetic-data construction. This is a single observation on the verification machine, not a throughput or latency guarantee; width, cardinality, estimator choice and concurrent load materially affect costs.

## Deployment and optional integrations

The final root image was built using Docker Desktop's Linux engine. Its image ID is `sha256:acb6746087c2c9e174315869eb3c3b87d6a88f2a215eaec639d44c1aca2f620c`. The `analytiq` container serves port 8080 as non-root user `analyst` (UID 10001), with `analytiq-data` mounted at `/app/data`. Verification used `MAX_WORKERS=1`; the application default remains 2.

Before replacing the intermediate v3 container, there were no active jobs. The replacement preserved exactly **24 users, 53 datasets, 19 runs, 3 experiments and 15 saved pipelines**. The combined browser tests subsequently created additional isolated demo workspaces and runs.

A restart after the browser suite preserved **30 users, 67 datasets, 25 runs, 6 experiments and 18 saved pipelines**. A digest of stored run configurations/results/events, dataset references and experiment records matched before/after restart. All registered dataset files and all completed-run reports/ZIPs/required pipelines were present. Existing signed sessions continued to retrieve completed analytical evidence, report/package downloads and experiment membership after restart.

Health reported:

```json
{"status":"ok","version":"3.0.0","llm_enabled":false,"demo_enabled":true}
```

Both `docker compose config -q` and `docker compose -f docker-compose.multi.yml config -q` passed. The multi-service PostgreSQL/API/worker/Nginx configuration was validated but not started during this verification.

LLM-available contract checks used mocked HTTP transport; no live OpenAI provider/key was tested. The deployed no-key deterministic path was exercised throughout the browser workflows.

## Earlier Agentation development feedback

The preceding UI verification on the same date exercised Agentation in development on port 5173 against the Docker API:

- Activated feedback mode, selected a sign-in workflow heading and added an annotation.
- Verified copied feedback included the note, CSS location and `src/pages/Auth.tsx` source location.
- Visually reviewed the annotation screenshot and observed no uncaught browser errors.
- Confirmed the production bundle excludes the development toolbar.

Agentation remains DEV-only in `src/main.tsx`. These annotation checks are historical UI results; they were not rerun as part of the final six-workflow autonomous-engine suite.

## Scope and remaining limits

This record establishes the checks above, not production certification or penetration/load-test coverage. Objective interpretation, target suitability and confidence remain transparent heuristic rules; exploratory associations/scenarios are not causal findings. Forecasts are bounded temporal baselines without uncertainty intervals. Search budgets/cancellation operate at recorded boundaries rather than hard interruption of a running estimator. Live-provider behavior, the multi-service runtime and deployment-specific TLS, quotas, backups and monitoring require separate operational validation. See `SECURITY.md`, `STATISTICS.md` and `REPRODUCIBILITY.md` for the complete analytical and deployment contracts.
