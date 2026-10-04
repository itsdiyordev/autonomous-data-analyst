"""Conservative leakage review: suspicious features remain visible and retained."""
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score


def detect_leakage(frame, target, profile):
    if not target or target not in frame:
        return {"target": target, "flags": [], "policy": "No predictive target was identified."}
    sample = frame.sample(min(5000, len(frame)), random_state=42)
    metadata = {column["name"]: column for column in profile["columns"]}
    flags = []
    prediction_time = next((name for name in frame if re.search(r"prediction_(time|date)|as_of|observation_date", name, re.I)), None)
    for name in frame.columns:
        if name == target:
            continue
        reasons = []
        evidence = {}
        if re.search(r"(^|_)(post|after|future|next|target|outcome)(_|$)|cancell?ation|churn_date|termination|refund_after", name, re.I):
            reasons.append("The feature name suggests outcome-derived or post-event information that may not exist at prediction time.")
        pair = sample[[name, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) >= 20 and not metadata.get(name, {}).get("possible_id"):
            if pd.api.types.is_numeric_dtype(pair[name]) and pd.api.types.is_numeric_dtype(pair[target]) and min(pair.nunique()) > 1:
                association = float(pair[name].corr(pair[target]))
                if np.isfinite(association) and abs(association) >= 0.97:
                    reasons.append("An extremely strong target correlation may indicate a target-derived measurement; domain review is required.")
                    evidence["correlation"] = association
            elif 2 <= pair[name].nunique() <= 20 and 2 <= pair[target].nunique() <= 20:
                table = pd.crosstab(pair[name].astype(str), pair[target].astype(str))
                if min(table.shape) >= 2:
                    chi, _, _, _ = chi2_contingency(table, correction=False)
                    value = float(np.sqrt(chi / (len(pair) * (min(table.shape) - 1))))
                    if value >= 0.98:
                        reasons.append("Near-perfect categorical association may encode the target; high predictiveness alone does not prove leakage.")
                        evidence["cramers_v"] = value
            elif pd.api.types.is_numeric_dtype(pair[name]) and pair[target].nunique() == 2:
                classes = sorted(pair[target].astype(str).unique())
                labels = (pair[target].astype(str) == classes[1]).astype(int)
                auc = float(roc_auc_score(labels, pair[name]))
                separation = max(auc, 1 - auc)
                if separation >= 0.995:
                    reasons.append("A single numeric feature almost perfectly separates the outcome; verify that its measurement precedes the event. This can also be a valid strong predictor.")
                    evidence["single_feature_auc_orientation_invariant"] = separation
        if prediction_time and name != prediction_time and metadata.get(name, {}).get("kind") == "datetime":
            event = pd.to_datetime(sample[name], errors="coerce", utc=True, format="mixed")
            reference = pd.to_datetime(sample[prediction_time], errors="coerce", utc=True, format="mixed")
            complete = event.notna() & reference.notna()
            fraction = float((event[complete] > reference[complete]).mean()) if complete.any() else 0
            if fraction > 0.1:
                reasons.append("Some feature timestamps occur after the recorded prediction/observation time.")
                evidence["after_prediction_fraction"] = fraction
        if reasons:
            flags.append({"feature": name, "reasons": reasons, "evidence": evidence, "sample_rows": len(pair), "action": "Verify availability before prediction; not automatically removed."})
    return {"target": target, "flags": flags, "policy": "Potential leakage is a review flag, not proof. Suspicious columns are not silently removed; structural feature exclusions are recorded separately."}
