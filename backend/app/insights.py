from .confidence import assess
from .evidence import Insight


def generate_insights(evidence):
    output = []
    for item in evidence:
        if item.status != "completed":
            continue
        values = item.values
        finding, why, severity, kind = None, "", "low", item.kind
        if item.kind == "data_quality":
            finding = f"The dataset contains {values['rows']:,} rows, {values['missing_cells']:,} missing cells, {values['duplicates']:,} exact duplicates and {values['invalid_values']:,} recorded invalid values."
            why, severity = "Data quality determines which analyses can be trusted and what needs review before deployment.", "high" if values["missing_pct"] > 10 or values["invalid_values"] else "medium"
        elif item.kind in {"association", "group_difference"}:
            if not values.get("significant"):
                continue
            effect, q = values["effect_size"], values["q_value"]
            finding = f"{item.title}: adjusted p = {q:.4g}, {values['effect_measure']} = {effect:.4g}; {values['practical_strength']} observed effect."
            if values["practical_strength"] == "negligible":
                finding += " Statistically detectable but practically weak under the recorded effect-size rule."
            why, severity = "Use the measured association to prioritize further investigation, not to infer causality.", "high" if values["practical_strength"] == "large" else "medium"
        elif item.kind == "temporal":
            change = values.get("latest_growth_pct")
            finding = f"{item.columns[1]} has a {values['trend']} descriptive trend across {values['periods']} {values['frequency']} periods."
            if change is not None:
                finding += f" Latest observed period change: {change:+.2f}%."
            why, severity, kind = "Separate partial periods, changing coverage and segment mix before attributing this change to a business cause.", "high" if change is not None and abs(change) >= 10 else "medium", "trend"
        elif item.kind == "anomaly":
            finding = f"Isolation Forest flagged {values['flagged_rows']} of {item.sample_rows} sampled rows ({values['flagged_pct']:.2f}%) for review."
            why, severity = "Unusual observations may be valid rare events, data errors or changes in behavior; all rows are retained.", "medium"
        elif item.kind == "leakage" and values.get("flags"):
            finding = f"Potential leakage review flagged {len(values['flags'])} columns: " + ", ".join(flag["feature"] for flag in values["flags"])
            why, severity, kind = "Confirm these features are available before the predicted event before trusting model performance.", "high", "risk"
        elif item.kind == "imbalance":
            if not values.get("severe"):
                continue
            finding = f"The smallest class accounts for {values['minority_share'] * 100:.2f}% of labeled records. Accuracy alone is not an appropriate selection criterion."
            why, severity, kind = "Review minority-class precision/recall, PR-AUC and balanced metrics before using predictions.", "high", "risk"
        elif item.kind == "model":
            finding = f"{values['model_name']} was selected using development {values['selection_metric']}; held-out {values['primary_metric']} = {values['test_score']:.4g}."
            why, severity = "Model scores describe this test sample; calibration, leakage, stability and expected error constrain use on new data.", "medium"
        if not finding:
            continue
        confidence, reasons = assess(item)
        output.append(Insight(id=f"i{len(output) + 1:03d}", type=kind, severity=severity, confidence=confidence,
                              finding=finding, why_it_matters=why, evidence_ids=[item.id], confidence_reasons=reasons,
                              limitations=item.limitations or ["Observational evidence does not establish causal impact."]))
    return sorted(output, key=lambda item: ({"high": 0, "medium": 1, "low": 2}[item.severity], {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[item.confidence]))[:16]
