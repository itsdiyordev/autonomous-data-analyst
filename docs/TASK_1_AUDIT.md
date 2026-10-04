# Task 1 — maintenance and security audit

## Scope and source of truth

Audited on October 4, 2026, starting from clean commit `d34c875` (`feat: expand autonomous analysis engine with statistical testing, evidence-backed insights, and data science workflows`). The earlier upgrade audit describes historical work; this document audits the implemented repository at the start of Task 1.

Reviewed backend routes, persistence, authentication/ownership, ingestion/profiling, planning/orchestration, statistics/hypotheses, temporal/anomaly/leakage checks, model search/evaluation, feature engineering/SHAP, evidence/findings/actions/reports, workers/cancellation, frontend routes/components/API handling, startup scripts, Docker/Compose/Nginx, dependency locks, tests/E2E, documentation and CI/CD presence. There are no checked-in GitHub Actions workflows or another CI/CD pipeline.

Task 1 retains React/FastAPI/SQLAlchemy, immutable Parquet datasets, the existing analytical/ML engines, bounded process workers and current UI. Changes correct existing contracts and failures. Release identity is now **3.0.1**; model-package schema remains **3.0**.

## Severity definitions

- **P0 — critical:** confirmed arbitrary execution, cross-account disclosure, destructive corruption or equivalent systemic failure.
- **P1 — high:** confirmed important workflow/evaluation failure, unsafe error/input handling or state/resource-control correctness gap.
- **P2 — medium:** bounded analytical assumptions, operational scaling/automation gaps and less severe consistency limitations.
- **P3 — low:** diagnostics, presentation and tooling maintenance.

No P0 issue was identified in this audit. The P1 findings below are fixed with focused changes and regression coverage.

## P1 findings and fixes

| ID | Problem and impact | Fix | Verification |
|---|---|---|---|
| T1-01 | Default validation errors reflected plaintext password inputs; unexpected API/job/candidate/SHAP failures returned inconsistent or raw exception text. | Structured HTTP `error` envelope, sanitized validation issues, safe expected analytical codes, generic unexpected messages, request IDs/security headers on errors, compatible legacy `detail` and run `error` text. | Password-redaction, injected API/job failure, actionable insufficient-data and browser invalid-input/recovery checks. |
| T1-02 | Content-Length checks could be bypassed by streamed JSON; multipart uploads were fully parsed/spooled before the endpoint's file-size check. | ASGI receive-byte limits before parsers: 2 MiB general bodies; configured upload limit + 2 MiB multipart allowance; existing per-file cap retained. | Direct ASGI multi-chunk JSON/multipart tests require HTTP 413 before all chunks are consumed; original oversized-file tests remain. |
| T1-03 | Read-then-write cancellation could overwrite concurrently completed/failed/cancelled runs. Progress/plan writes and late worker exceptions could modify terminal state. | Conditional SQL transitions; worker writes/failures require `running`; cancellation requires queued/running; publication guard retained. | Deterministic stale-read cancellation reproductions, terminal-write and late-failure tests. |
| T1-04 | Process-pool submission failures or transient SQL errors could terminate the dispatcher and strand later queued work. | Retained failed futures, pool replacement, safe interrupted-job errors and retryable database handling. | Controlled broken-pool and transient-database tests verify subsequent jobs complete. |
| T1-05 | Chronological row cuts/CV could split identical timestamps across evaluation partitions. Sorting also overwrote/dropped a real `__sort_time` column. | Stable index-based sorting and timestamp-atomic holdout/CV boundaries; insufficient distinct timestamps use explicit errors or separate-validation fallback. | Duplicate-timestamp split/fold separation and reserved-column retention tests; original group/time tests pass. |
| T1-06 | Generated date features silently overwrote source fields such as `date__year`. | Reject colliding feature schemas with an actionable error; portable transformer remains self-contained. | Collision rejection and independent exported date-inference tests. |
| T1-07 | An explicit supervised task could lose its target when the objective mentioned segments. Explicit clustering could become a forecast intent or be rejected for an irrelevant constant target. | Explicit task contract takes precedence over conflicting substring heuristics; clustering is target-free. | Explicit classification/regression/segmentation contradiction tests. |
| T1-08 | Explicit numeric class labels could be treated as continuous targets in statistical correlations and anomaly distances. | Existing task contract supplies nominal target roles to statistics; class codes are excluded from numeric anomaly features. | A real 18-class autonomous run verifies group tests rather than ordinal correlations and excludes class-code anomaly distances. |
| T1-09 | Deleting a dataset referenced by an empty experiment raised an unhandled foreign-key error. | Check both run and experiment references; concurrent FK conflicts roll back and return HTTP 409. | Empty-experiment deletion rejection preserves the dataset preview. |
| T1-10 | Expired limiter keys were never removed, permanently exhausting capacity after enough clients. Active-run quota check/insertion lacked a serialization boundary. | Reclaim expired capacity; transactionally lock the account before checking/inserting runs; include Retry-After for request exhaustion. | Expired 10,000-key capacity recovery and eight concurrent create requests (three accepted/five limited). |
| T1-11 | Blank normalized names could break frontend avatar rendering; prediction/scenario dictionaries accepted nested or non-finite values unsupported by tabular inference. | Validate trimmed names, safely present legacy blank names, and restrict prediction/scenario fields to finite scalar values. | Blank-name/legacy-display and nested/non-finite contract tests; existing valid inference/export workflows pass. |
| T1-12 | JWT verification checked expiry only when supplied, allowing a correctly signed token without the advertised session expiry. | Require `sub`, `exp`, `iat` while retaining HS256 signature/expiry verification. | Missing-expiry, expired and tampered-token tests. |

Re-run parent metadata is also committed with the queued run rather than appended after it becomes visible to a worker. The reproducibility hash now uses a single patch-release engine identity so corrected analyses are distinguishable from prior results.

## Baseline and verification

| Check | Before fixes | After fixes |
|---|---|---|
| Existing backend suite | 57 passed, 3 warnings, 55.95 seconds | 92 passed, 3 warnings, 39.65 seconds |
| Ruff | Passed | Passed |
| Frontend TypeScript | Passed | Passed |
| Frontend production build | Passed; 746 modules | Passed; 746 modules |
| Existing Chromium flows | 6 passed in 2.4 minutes after environment recovery | 6 passed in 1.8 minutes against the fixed image, including invalid-input recovery |
| Docker | Baseline image built | Fixed image built with frozen Python lock and `npm ci` |
| Compose | Both definitions inspected | Both configuration validations passed |
| npm audit, production and full dependency sets | No known vulnerabilities reported | Same dependency lock |
| pip-audit, installed Python environment | No known vulnerabilities reported | Same scientific dependency versions |

Initial new regression checks exposed **22 failures in 27 cases**. Separate dispatcher and planner/statistical reproductions exposed another two and four failures respectively. The final suite adds **35 focused cases**, including parametrized cases and expected-error/legacy compatibility checks, while retaining all 57 existing cases.

The first browser attempt stalled with exhausted 1 GiB swap on a 3.5 GiB WSL machine. Moving the 658 MiB browser cache off RAM-backed `/tmp` resolved resource pressure. A WSL restart then cleared temporary browser libraries; restoring them to the Linux disk cache resolved a launch-only retry failure. The recovered baseline passed all six scenarios. These failures were environment failures, not hidden passing results. Final browser/deployment results are recorded separately in `VERIFICATION.md`.

Warnings are one Starlette/httpx TestClient deprecation and two bounded MLP CV convergence warnings. Dependency scans are advisory-database snapshots, not proof of the absence of all vulnerabilities.

The verified image was deployed on port 8080 while retaining `analytiq-data`, with no active jobs before replacement. All **30 users, 67 datasets, 25 runs, 6 experiments and 18 pipelines** survived. A digest of every row in the four application tables matched before/after replacement; all required dataset/report/package/pipeline files were present. The oldest saved model still predicted and served its report/package through the owned API, and a legacy failed run exposed a safe structured diagnostic. The container is running/healthy as non-root `analyst`, with one worker for this verification machine.

## Security baseline

Verified by source review and automated checks:

- Ownership checks precede dataset/run/model/experiment access; authenticated IDs do not control filesystem paths or SQL strings.
- Static and artifact paths must stay under their trusted roots; filenames are normalized across Windows/POSIX path syntax.
- CSV/Excel allowlists, row/column caps, per-file/streaming request caps and ZIP traversal/expansion/encryption/ratio checks are enforced. XML parsing uses `defusedxml`; archives are not extracted into caller-controlled directories.
- Uploaded `.pkl`, `.pickle` and `.joblib` files are rejected; regression tests ensure they never call `joblib.load`. Only owned, completed server-generated pipelines are loaded.
- Argon2 credentials, required signed JWT claims, expiry, bounded request categories and serialized active-run limits remain in place.
- Untrusted objectives/dataset text cannot execute code; the optional LLM may only select existing computed evidence IDs. Invented facts/IDs, extra fields and provider failures are rejected or fall back.
- React text rendering/report escaping/CSP remain; public errors omit inputs, tracebacks and internal exception text. Detailed debugging is confined to server logs.

## Remaining P2/P3 issues

| ID | Severity | Remaining issue / next action |
|---|---|---|
| T1-R01 | P2 | No checked-in CI/CD pipeline. Automate the documented lint/tests/build/security checks and release gates in a later task. |
| T1-R02 | P2 | Auto-mode intent/target/frequency/aggregation interpretation remains English-oriented and heuristic. Ambiguous objectives need later semantic work; explicit contract contradictions are fixed here. |
| T1-R03 | P2 | Statistical screening, effect thresholds and confidence are bounded exploratory heuristics; dependence, sparse tables, sample selection and missingness require caution/replication. Forecast baselines lack uncertainty intervals. |
| T1-R04 | P2 | Request limiting remains per-process/IP. The optional Nginx deployment requires an explicit trusted-proxy/client-IP policy and deployment-wide limits before multi-instance operation. |
| T1-R05 | P2 | Upload profiling/quality review are synchronous request-thread work. Wide/high-cardinality load, sustained concurrency, storage/demo quotas and operational monitoring need separate validation. |
| T1-R06 | P2 | Cancellation/search budgets are cooperative, not estimator hard deadlines. Stale recovery still uses stage updates and a 30-minute cutoff rather than dedicated worker heartbeats. |
| T1-R07 | P2 | PostgreSQL/multi-service configuration is validated but its runtime, migrations, backups/restore, TLS and hostile load/penetration tests remain separate operational work. The optional Nginx 1.27 image lifecycle and OS/container-image advisory scanning need review; Python/npm scans do not cover them. Startup secret-file initialization across unsupported concurrent first-start instances also needs hardening. |
| T1-R08 | P2 | Database/file operations are separate; failed registration/cancelled publication can leave orphan artifacts. File cleanup and profile/result schema migrations remain manual. |
| T1-R09 | P3 | TestClient deprecation and bounded MLP convergence warnings remain; address tooling compatibility/convergence reporting in routine maintenance. |
| T1-R10 | P3 | Some dashboard/model-studio labels still use legacy training wording; validation details are available through `ApiError` but not rendered per form field. |
| T1-R11 | P2 | SHAP transformed-name aggregation uses longest-prefix matching. Overlapping raw field names/category tokens can make lineage ambiguous; replace this heuristic with explicit transformer lineage in later explainability work. Raw-column permutation importance is computed independently. |

## Task 2 integration risks

1. Preserve the structured HTTP envelope and compatibility fields. Background runs retain string `error` and add typed `failure`; use the safe analytical error contract rather than returning `str(exception)`.
2. Preserve typed step kinds, ownership checks, deterministic numerical computation, immutable data and conditional terminal-state guards when changing planning.
3. Auto mode still has ambiguous substring heuristics. A future intent resolver must not override explicit tasks/targets into inconsistent ML contracts.
4. Version analytical behavior through `app/version.py` and stored reproducibility metadata. Older chronological results may contain duplicate-timestamp leakage; keep their immutable artifacts and rerun them for corrected evaluation.
5. Numeric class codes are nominal outcomes. Future semantic work must preserve that explicit type contract and distinguish observed measurements from label identities.
6. Model-package schema 3.0 and legacy ML-mode requests remain supported. Existing stored plans/results/profiles need an intentional migration/version policy for future incompatible changes.
7. Keep heavyweight verification sequential on small WSL machines and cache Python/browser dependencies on the Linux disk filesystem.

Detailed final commands and deployment evidence: [VERIFICATION.md](VERIFICATION.md). Current implementation/data flow: [ARCHITECTURE.md](ARCHITECTURE.md).
