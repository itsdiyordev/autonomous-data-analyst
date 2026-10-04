"""Confidence is a transparent rule assessment, not a probability or LLM opinion."""


def assess(item):
    values = item.values
    reasons = [f"Computed from {item.sample_rows} observations using {item.method}."]
    if item.kind == "data_quality":
        return "HIGH", reasons + ["Direct counts over the registered dataset; confidence refers to observed quality, not downstream conclusions."]
    if item.kind in {"association", "group_difference"}:
        q = values.get("q_value", 1)
        strength = values.get("practical_strength", "negligible")
        reasons += [f"Adjusted p-value: {q:.4g}; effect magnitude: {strength}."]
        uncertain = values.get("sparse_expected", False) or any("Temporal dependence" in note or "Duplicate records" in note for note in item.limitations)
        if q < 0.01 and strength in {"moderate", "large"} and item.sample_rows >= 100 and not uncertain:
            return "HIGH", reasons + ["Adequate sample, nontrivial effect and multiplicity-adjusted evidence; independent confirmation is still required."]
        if q < 0.05 and strength != "negligible" and item.sample_rows >= 30 and not values.get("sparse_expected", False):
            return "MEDIUM", reasons + ["Evidence is exploratory or the assumptions/sample limit confidence."]
        return "LOW", reasons + ["Small/weak effects, inadequate evidence or assumption concerns prevent a strong conclusion."]
    if item.kind == "model":
        stability = values.get("stability", {}).get("status")
        if values.get("leakage_flags", 0):
            return "LOW", reasons + ["Potential leakage requires domain review before interpreting predictive performance."]
        if stability == "stable" and item.sample_rows >= 100 and values.get("beats_baseline"):
            return "HIGH", reasons + ["Stable development folds, independent held-out evaluation and improvement over the baseline."]
        return "MEDIUM" if item.sample_rows >= 40 else "LOW", reasons + ["Limited test sample, variable folds or no clear baseline improvement."]
    if item.kind == "temporal":
        complete = values.get("missing_periods", 1) == 0 and values.get("periods", 0) >= 24
        return "MEDIUM" if complete else "LOW", reasons + ["Temporal patterns are descriptive; partial periods and changing observation counts may affect interpretation."]
    return "LOW", reasons + ["Algorithmic flags or domain assumptions require manual review and independent confirmation."]
