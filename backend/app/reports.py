"""Every report sentence containing a quantity is rendered from computed evidence."""
import json


def analytical_report(name, objective, analysis, model=None):
    quality = next(item for item in analysis["evidence"] if item["kind"] == "data_quality")["values"]
    summary = "\n".join(insight["finding"] for insight in analysis["insights"][:3]) or "Exploratory analysis completed; inspect evidence and limitations before drawing a strong conclusion."
    sections = [
        ("Executive Summary", summary),
        ("Dataset Overview", f"{quality['rows']:,} rows; {quality['columns']} columns; quality {quality['quality_score']}/100; {quality['missing_cells']:,} missing cells; {quality['duplicates']:,} exact duplicate records. Outliers were retained."),
        ("Analytical Objective", objective),
        ("Analysis Plan", "\n".join(f"{step['title']} [{step['status']}]: {step['reason']}" for step in analysis["plan"]["steps"])),
        ("Key Findings", "\n\n".join(f"{item['finding']}\nConfidence: {item['confidence']}. {' '.join(item['confidence_reasons'])}\nEvidence: {', '.join(item['evidence_ids'])}" for item in analysis["insights"])),
        ("Statistical Evidence", json.dumps([item for item in analysis["evidence"] if item["kind"] in {"association", "group_difference"}], indent=2)),
        ("Hypotheses", json.dumps(analysis["hypotheses"], indent=2)),
        ("ML Results", (f"Model: {model['model_name']}. {model['selection_note']}\nCV: {json.dumps(model['cross_validation'])}\nTest: {json.dumps(model['metrics'])}\nStability: {json.dumps(model.get('model_stability'))}\nCalibration: {json.dumps(model.get('calibration'))}" if model else analysis["ml_decision"]["reason"])),
        ("Explainability", json.dumps({"permutation_importance": model.get("feature_importance", []), "shap": model.get("shap", {})}, indent=2) if model else "No predictive model was required; relationships are supported by the recorded statistical evidence."),
        ("Anomalies", json.dumps([item for item in analysis["evidence"] if item["kind"] == "anomaly"], indent=2)),
        ("Temporal Analysis and Forecast", json.dumps([item for item in analysis["evidence"] if item["kind"] == "temporal"], indent=2)),
        ("Recommendations", "\n\n".join(f"{item['action']}\nEvidence: {', '.join(item['evidence_ids'])}. {item['limitation']}" for item in analysis["recommendations"])),
        ("Limitations", "\n".join(analysis["limitations"])),
        ("Reproducibility", json.dumps(analysis["reproducibility"], indent=2)),
    ]
    analysis["executive_summary"] = summary
    analysis["report_sections"] = [{"title": title, "text": text} for title, text in sections]
    return f"FINAL ANALYTICAL REPORT — {name}\n\n" + "\n\n".join(f"{title.upper()}\n{text}" for title, text in sections)
