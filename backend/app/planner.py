"""Objective-driven, deterministic planning; descriptive objectives never force ML."""
import re

from .evidence import AnalysisPlan, AnalysisStep, MLDecision
from .problem import rank_targets


def create_plan(profile, objective, target=None, requested_task="auto", config=None):
    config = config or {}
    columns = {column["name"]: column for column in profile["columns"]}
    if target and target not in columns:
        raise ValueError("The selected target is not in this dataset.")
    text = objective.lower()
    forecast = requested_task != "clustering" and bool(re.search(r"forecast|future|next (month|week|day|period)", text))
    prediction = bool(re.search(r"predict|classif|estimate|model\b", text))
    segments = requested_task == "clustering" or requested_task == "auto" and bool(re.search(r"cluster|segment|similar (customers|records|rows)", text))
    quality_only = bool(re.search(r"missing|null|quality|duplicate|invalid|clean", text)) and not re.search(r"trend|relation|pattern|why|driver|predict|forecast|segment|compare|growth", text)
    intent = "forecast" if forecast else "segmentation" if segments else "predictive" if prediction else "quality" if quality_only else "diagnostic" if re.search(r"why|decreas|increas|driver|explain|difference", text) else "descriptive"
    ranked = rank_targets(profile, objective)
    if segments:
        target = None
    if not segments and not target and ranked and ranked[0].score >= 0.75 and (len(ranked) == 1 or ranked[0].score - ranked[1].score >= 0.1):
        target = ranked[0].name
    numeric = [column["name"] for column in columns.values() if column["kind"] == "numeric" and not column.get("possible_id") and column["unique"] > 1]
    categories = [column["name"] for column in columns.values() if column["kind"] == "categorical" and 2 <= column["unique"] <= 20 and not column.get("possible_id")]
    dates = [column["name"] for column in columns.values() if column["kind"] == "datetime"]
    task = "clustering" if segments else (next((item.task for item in ranked if item.name == target), "regression" if target in numeric else "classification") if target and prediction else "descriptive")
    if requested_task in {"regression", "classification"} and target:
        task, prediction = requested_task, True
    if task == "classification" and target:
        # A supplied classification task defines a nominal outcome even when its
        # stored labels are numeric. Do not infer distances between label codes.
        numeric = [name for name in numeric if name != target]
        if target not in categories and 2 <= columns[target]["unique"] <= 20:
            categories.insert(0, target)
    use_ml = segments or (prediction and target is not None and not forecast)
    reason = "Segmentation was requested; compare unsupervised group models." if segments else "A predictive objective and a suitable target were identified." if use_ml else "The forecasting objective uses chronological, evaluated time-series baselines rather than tabular AutoML." if forecast else "Machine learning was not used because the objective is descriptive/diagnostic or no reliable predictive target was identified."
    limitations = []
    if forecast and (not dates or target not in numeric):
        reason = "No reliable date/quantitative outcome combination was identified; forecasting was not attempted. Exploratory evidence remains available."
        limitations.append("Specify a valid date column and an unambiguous numeric forecast outcome.")
    if use_ml and profile["rows"] - profile.get("duplicates", 0) < 40:
        use_ml = False
        limitations.append("Predictive training requires at least 40 usable distinct rows; exploratory analysis remains available.")
        reason = limitations[-1]
    if use_ml and target and (columns[target]["unique"] < 2 or task == "classification" and (columns[target]["unique"] > 20 or columns[target].get("usable_minimum_class_count", columns[target].get("minimum_class_count", 8)) < 8)):
        use_ml = False
        reason = "The target is constant or has insufficient supported class examples; exploratory analysis is retained instead of forcing unreliable ML."
        limitations.append(reason)
    if use_ml and target and columns[target].get("usable_non_missing_rows", profile["rows"]) < 40:
        use_ml = False
        reason = "Fewer than 40 distinct rows have a usable target; exploratory analysis is retained instead of forcing ML."
        limitations.append(reason)
    if use_ml and task == "regression" and columns[target]["kind"] != "numeric":
        use_ml = False
        reason = "Regression requires a varying numeric outcome. The selected outcome is not numeric; review its type or use an appropriate classification objective."
        limitations.append(reason)
    usable_features = [column for column in columns.values() if (column["name"] != target or segments) and not column.get("constant") and not column.get("possible_id") and not (column["kind"] == "categorical" and column["unique"] > 100)]
    width = sum(10 if column["kind"] == "datetime" else 2 if column["kind"] == "numeric" else min(24, column["unique"]) for column in usable_features)
    if use_ml and (not usable_features or len(usable_features) > 100 or width > 1200 or profile["rows"] * width * 8 > 512 * 1024 * 1024):
        use_ml = False
        reason = "The supported feature/memory contract is not suitable for this model search; exploratory evidence is retained instead of forcing resource-unsafe ML."
        limitations.append(reason)
    if use_ml and any(column.get("extreme_magnitude") for column in columns.values()):
        use_ml = False
        reason = "Extreme finite numeric magnitudes exceed the stable predictive contract; inspect scaled descriptive evidence and measurement units first."
        limitations.append(reason)
    if (prediction or forecast) and not target:
        limitations.append("No reliable target detected. Specify an outcome or run exploratory analysis first.")
    decision = MLDecision(use_ml=use_ml, task=task if use_ml else "descriptive", target=target if use_ml and not segments else None, reason=reason, limitations=limitations)
    steps = []

    def add(kind, title, why, selected, outputs):
        steps.append(AnalysisStep(id=f"s{len(steps) + 1:02d}", kind=kind, title=title, reason=why, columns=selected, expected_outputs=outputs))

    add("profile", "Dataset understanding", "All analyses need a reproducible description of the original data.", list(columns), ["schema", "column roles", "ranked target candidates"])
    add("quality", "Data quality assessment", "Missingness, invalid values and duplicate records affect reliability.", list(columns), ["quality evidence", "actionable issues"])
    if dates and numeric and not quality_only:
        measure = target if target in numeric else numeric[0]
        add("temporal", "Trend and temporal analysis", "A valid date column and quantitative measurements support time-based analysis" + ("; the objective concerns change or forecasting." if intent in {"diagnostic", "forecast"} else "."), [dates[0], measure], ["trend", "missing periods", "growth", "anomalies", "evaluated forecast baselines"])
    if not quality_only and (len(numeric) >= 2 or len(categories) >= 2 or (numeric and categories)):
        selected = ([target] if target else []) + numeric[:8] + categories[:4]
        add("associations", "Relationships and association matrix", "Variable types determine numeric, categorical or mixed association methods.", list(dict.fromkeys(selected)), ["Pearson/Spearman", "Cramér's V", "mixed associations"])
        if categories and numeric:
            add("group_comparison", "Compare meaningful groups", "Categorical groups and quantitative outcomes allow segment comparisons.", [categories[0], target if target in numeric else numeric[0]], ["group sizes", "means and medians"])
        add("statistics", "Statistical testing and effect sizes", "Observed relationships need uncertainty, assumptions and practical magnitude.", list(dict.fromkeys(selected)), ["tests", "effect sizes", "adjusted p-values"])
        add("hypotheses", "Test exploratory hypotheses", "Generate explicit hypotheses from the computed observations and connect their tests.", list(dict.fromkeys(selected)), ["hypotheses", "test conclusions", "exploratory limitations"])
    if numeric and not quality_only:
        add("anomalies", "Outlier and anomaly analysis", "Quantitative measurements support IQR and multivariate anomaly review; flagged rows are retained.", numeric[:12], ["IQR evidence", "Isolation Forest anomalies"])
    if target and not quality_only and not segments:
        add("leakage", "Potential leakage review", "Outcome-associated fields may be unavailable at prediction time; suspicious columns are flagged, not silently removed.", [target], ["semantic/association/temporal leakage flags"])
    if task == "classification":
        add("imbalance", "Class imbalance analysis", "Minority outcomes require suitable metrics and class-weight context.", [target], ["class distribution", "metric policy"])
    add("ml", "ML decision and model comparison", decision.reason, [target] if target else [], ["decision", "baseline", "CV", "held-out evaluation"] if use_ml else ["recorded reason for skipping ML"])
    if use_ml:
        add("explainability", "Explain and evaluate the model", "Predictions need held-out errors, stability, calibration context and supported feature contributions.", [], ["SHAP", "permutation importance", "model limitations"])
    add("insights", "Evidence-backed findings", "Rank computed findings and assess confidence using transparent rules.", [], ["findings", "confidence reasons", "limitations"])
    add("recommendations", "Recommended next steps", "Actions must reference findings and cannot assert unmeasured causal effects.", [], ["evidence-linked actions"])
    add("report", "Final analytical report", "Preserve the objective, decisions, evidence and reproducibility record together.", [], ["executive summary", "HTML report", "JSON evidence"])
    return AnalysisPlan(objective=objective, intent=intent, task=decision.task, target=target, target_candidates=ranked, steps=steps, ml_decision=decision, reasons=[f"Detected {len(dates)} date, {len(numeric)} usable numeric and {len(categories)} bounded categorical columns.", decision.reason], expected_outputs=list(dict.fromkeys(output for step in steps for output in step.expected_outputs)))
