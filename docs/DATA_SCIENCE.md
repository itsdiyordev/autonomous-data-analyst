# Data science implementation

## Academic concepts and concrete implementations

| Concept | Implementation | Inspectable product evidence |
|---|---|---|
| Descriptive analytics / EDA | `data.py`, `planner.py`, `statistics.py` | Distributions, quality center, counts, grouped summaries, association records |
| Diagnostic analytics | `time_series.py`, `statistics.py`, `hypothesis.py` | Trends, group differences, exploratory hypotheses, assumptions and confounding limitations |
| Predictive analytics | `ml.py`, `model_assessment.py`, `time_series.py` | Development-only selection, held-out errors, calibration, evaluated forecast baselines |
| Prescriptive analytics | `recommendations.py`, scenario API | Evidence-linked next steps and explicitly noncausal changed-input model scenarios |
| Statistical inference | `statistics.py`, `confidence.py` | Tests, p/q-values, confidence intervals, effect sizes and transparent confidence rules |
| Explainable AI | `explain.py` | SHAP, held-out permutation importance, sampled supported directions and error examples |
| Data quality | `data.py` | Missingness, duplicates, invalid values, column roles, entropy, dates and outliers |
| Anomaly detection | `anomalies.py`, `time_series.py` | Isolation Forest flags, robust driver deviations and temporal residual flags |
| Reproducibility | `reproducibility.py`, immutable datasets, re-run API | Hashes, plan/configuration, seed, versions, artifacts and experiment comparisons |

## Profiling

Profile schema version 4 preserves existing keys and adds:

- Dataset memory usage, duplicate percentage, type ratios, constant/near-constant columns, potential IDs, high cardinality, free text and ranked targets.
- Numeric mean/median/SD/variance/skewness/kurtosis, range, quantiles, zero/negative percentages, recorded infinities and IQR bounds/counts.
- Categorical entropy, dominant category/share, rare-category percentage, distributions and usable class counts after exact duplicate handling.
- Date range, invalid/duplicate dates, frequency/regularity, missing-period estimates, future timestamps and suitability flags.

Near-constant means a dominant observed value share of at least 95%. High cardinality means more than 100 categorical values. ID/free-text flags are heuristic and require domain confirmation. The quality score remains `100 − missing-cell % − duplicate-row %`; it is not a comprehensive analytical reliability score.

Registered Parquet data preserves measured values with documented normalization: column names are trimmed and non-finite numeric inputs are recorded and represented as missing values. Potential outliers remain present. Extreme magnitudes use scaled profiling; numerically unrepresentable variance/bounds are null rather than fabricated.

## Supervised and unsupervised ML

Existing baseline, linear, decision-tree, random-forest, gradient-boosting and neural families remain supported. K-Means and diagonal Gaussian mixtures support segmentation; two-group K-Means supplies the clustering baseline.

Eligibility records development rows, estimated encoded width and class representation. Neural candidates require at least 100 search rows, at most 256 encoded inputs and adequate class counts. Dense ensemble/clustering bounds avoid unsuitable high-dimensional workloads. The baseline remains eligible. Search budgets are checked between candidates, not hard process deadlines.

Imputation/encoding/scaling/date engineering are inside each CV pipeline. Selection uses binary positive-outcome PR-AUC, multiclass macro F1, regression RMSE or clustering silhouette. Classification defaults to the minority outcome **among usable labeled rows** as the positive class; an explicit API label can override it. Original and usable class distributions are stored. Suitable classifiers use class weights; no automatic oversampling is performed.

Group/time splits use matching folds. All target classes must be present in every outer split. If group/time CV lacks classes, a recorded separate-validation fallback is used. The selected model is refitted on development data before test evaluation. Candidate test scores are not used for selection; only the selected model and a baseline receive final test context.

CV SD/mean stability is a stated heuristic, not a confidence interval. Binary calibration records Brier score, ten-bin reliability and ECE; multiclass Brier uses the sum-of-squares definition. No calibration model or threshold is fitted on test data. Threshold tradeoffs use the separate validation pipeline before final refitting.

Feature importance does not imply direction. Numeric direction is reported only when sampled values and their SHAP contributions support a descriptive Spearman relationship. It is small-sample model evidence, not a monotonic or causal guarantee. Prediction forms/API/exports include evaluation context; scenarios are explicitly **MODEL-BASED SCENARIO**, never a causal effect.

## Temporal and anomaly analysis

Calendar buckets are bounded and selected from observed intervals/span. Additive-looking sales/revenue/quantity/profit/volume measurements use sums; others use means. The aggregation choice is recorded and needs domain review. Missing periods are displayed, not silently interpolated.

Temporal evidence includes descriptive slope, latest-period growth, rolling averages, detrended seasonal-lag autocorrelation, robust residual anomalies and exploratory change-point candidates. Partial periods and changing row coverage can affect totals.

Forecasting compares naive, three-period moving average and fixed-alpha exponential smoothing. A chronological 60/20/20 partition separates training, selection validation and winner-only test evaluation. Evaluation is rolling one-step; the final six-period iterative forecast does not inherit a guaranteed six-step error bound. Forecast uncertainty intervals are not implemented.

Isolation Forest uses fixed seed, 80 trees, bounded rows and numeric dimensions. Scaling precedes median imputation to stabilize large magnitudes. Unusable sampled dimensions are explicitly recorded. Flags are algorithmic anomalies, not proven data errors; all original rows remain intact.

## Limitations

Supported ML requires at least 40 distinct usable rows, 1–100 source features and appropriate target variation/classes. The dense encoded contract is bounded to 1,200 inputs and approximately 512 MiB. Strong leakage flags require availability review and are not silently excluded. There is no universal causal inference, production drift monitoring, fairness certification or guaranteed prediction interval.
