# Autonomous analysis engine

Analytiq supports both explicit model training and objective-driven analytical investigations. The latter can finish successfully without training a model.

## Execution contract

```text
User objective + immutable dataset/profile
  → typed deterministic plan
  → selected EDA/statistical/temporal/anomaly checks
  → exploratory hypotheses and evidence
  → explicit ML decision
  → optional existing AutoML pipeline
  → evidence-backed insights + rule-based confidence
  → recommendations + computed report
  → optional LLM ordering of existing narrative statements
```

`evidence.py` defines `AnalysisPlan`, `AnalysisStep`, `MLDecision`, `Evidence`, `Insight`, and `Recommendation`. Step kinds are closed typed choices; unsupported executable analyses cannot be injected through user text. Evidence IDs link findings, hypotheses and actions to exact methods, values, sample sizes, assumptions and limitations.

## Planner decisions

`planner.create_plan` combines objective intent, column types/roles, target suitability, class counts after duplicate handling, usable row counts, date availability and feature/memory bounds.

- **Quality-only question:** profile and quality checks, an explicit no-ML decision, findings/actions/report. Unrelated statistical/time-series workloads are not run.
- **Diagnostic sales question:** when supported by the schema, temporal trends, numeric/mixed associations, group comparisons, statistical tests, hypotheses and anomalies. Predictive ML is not forced.
- **Predict churn:** ranked or explicit categorical target, imbalance/leakage review, baseline and eligible supervised models, CV and explanations.
- **Find segments:** clustering without a predictive target, group evaluation and profiles. A supplied analytical outcome does not become a cluster label.
- **Forecast future sales:** evaluated temporal baselines rather than random-split tabular AutoML. Missing dates or an ambiguous numeric outcome produce an explicit limitation.
- **Insufficient data, constant/rare targets, excessive encoded width or numeric scale:** retain exploratory analysis and explain why supported ML is unsuitable.

Target ranking uses column type/variation, semantic outcome names, objective matches and missingness. Scores are **heuristic suitability scores**, not calibrated probabilities. Automatic target selection requires score ≥ 0.75 and a margin ≥ 0.10 over the next candidate. Ambiguity does not force a target.

## Conditional orchestration

`orchestrator.execute_run` dispatches legacy ML runs to the existing trainer. Autonomous runs execute only their stored plan. Steps save running/completed/skipped state and append execution events containing reasons, step IDs and evidence references. Cancellation is checked at stage boundaries and publication remains conditional on the run still being `running`.

Statistical pair work is bounded to 24 planned pairs and 5,000 sampled rows. Isolation Forest uses up to 10,000 rows. Temporal analysis requires adequate distinct dates; forecasting additionally requires 24 complete aggregated periods. These choices are visible in evidence, not hidden performance shortcuts.

## Findings and actions

`insights.py` converts computed evidence into typed findings; it never asks an LLM to invent a result. `confidence.py` supplies HIGH/MEDIUM/LOW with explicit reasons. `recommendations.py` maps findings to bounded investigative/validation actions, including evidence references and observational limitations.

The frontend provides Objective & plan, Execution, Findings, Statistics & evidence, ML & explainability, Recommendations and Report views. Findings start with an executive result and reveal technical evidence progressively. Applicable runs link to the preserved model studio, predictions and model-based scenarios.

## Limits of autonomy

Intent interpretation is deterministic, English-oriented and heuristic. It is not general human analytical reasoning. The engine supports bounded tabular analyses, not arbitrary scientific domains, NLP feature modeling or arbitrary user-defined code. Confidence is not a probability of truth; observational findings require domain judgment and independent validation.
