# Upgrade audit — October 4, 2026

The complete tracked application, tests, configuration, deployment, and documentation were inspected before implementation. The working tree was clean; the existing repository remote is the requested GitHub repository.

## Implementation map

| Area | Existing implementation | Gap / required change | Regression boundary |
|---|---|---|---|
| API and identity | FastAPI, Argon2, JWT, ownership checks, bounded uploads | Owned plan previews, scenarios, re-runs, experiments | Keep existing routes and training request semantics |
| Persistence | Immutable Parquet datasets; SQLAlchemy run configuration, events, results | Typed evidence and plans in existing JSON fields; additive experiment table | No destructive migration of existing users, datasets, or runs |
| Execution | Atomic job claim; isolated processes; cancellation and stale recovery | Conditional analytical orchestrator before optional AutoML | Preserve claim/cancellation/publication guards |
| Profiling | Missingness, duplicates, IQR, sampled histograms and Pearson matrix | Finite-value robustness, column roles, dates, entropy, target ranking | Retain existing profile keys and explicitly report transformations |
| Task detection | Explicit target or simple name match; otherwise clustering | Objective intent and ranked heuristic candidates; allow descriptive analyses | Legacy ML mode remains available |
| ML | Six supervised families, clustering, leakage-safe folds, held-out testing | Eligibility, imbalance context, calibration, stability, timing, scenarios | Selection never uses test scores; saved feature transformer stays portable |
| Analytics | Profiling and model diagnostics | Typed statistical tests/effects, multiplicity correction, exploratory hypotheses, time series and anomalies | Sampling and observational limitations must be visible |
| Explanation | SHAP, permutation importance, free-text optional LLM | Evidence-backed direction and deterministic confidence/recommendations; restricted LLM evidence selection | No numerical claim comes from generated LLM text |
| Product | Guided model wizard, model studio, explorer, reports | Objective → plan → execution → findings/evidence → recommendations; experiments/re-run | Retain training, charts, predictions, exports, themes and responsive navigation |
| Security | SQL ownership isolation, static path checks, server-generated storage IDs | Archive expansion/path checks, artifact containment, bounded request controls, LLM output contract | Never accept uploaded model pickle or executable code |

## Engineering choices

- Keep React + FastAPI + SQLAlchemy + bounded Python workers; add no distributed infrastructure.
- Use explicit typed evidence IDs to connect tests, hypotheses, insights, confidence and actions.
- Statistical results are exploratory: pairwise complete cases, stated assumptions, effect sizes, and Benjamini–Hochberg adjustment. Correlation is not causation.
- Temporal analysis uses chronological baseline evaluation, records missing/duplicate periods, and suppresses unsupported forecasts.
- Immutable dataset hashes, stored plans/configuration, fixed seeds and runtime versions support equivalent analytical re-runs; timestamps and runtime need not be identical.
- Optional LLM assistance may select/reorder precomputed narrative blocks. Unsupported generated facts are rejected rather than becoming report evidence.
- Security/analytical tests accompany functionality; existing tests remain intact.
