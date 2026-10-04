import re

from .data import infer_task
from .evidence import TargetCandidate


def rank_targets(profile, objective=""):
    """Transparent suitability ranking; scores are deliberately not called probabilities."""
    text = re.sub(r"[^a-z0-9]+", " ", objective.lower())
    candidates = []
    semantic = {"target", "label", "outcome", "sales", "revenue", "profit", "price", "churn", "demand", "risk"}
    for column in profile["columns"]:
        if column.get("constant") or column.get("possible_id") or column["unique"] < 2 or column["kind"] == "datetime":
            continue
        name = column["name"]
        tokens = set(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())
        normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())
        match = bool(normalized and re.search(r"\b" + re.escape(normalized) + r"\b", text)) or bool(tokens & set(text.split()) & semantic)
        score, reasons = 0.2, []
        if column["kind"] == "numeric":
            score += 0.15
            reasons.append("Varying numeric measurements can support quantitative analysis.")
        elif column["unique"] <= 20:
            score += 0.1
            reasons.append("Repeated categories can support a classification outcome.")
        else:
            continue
        if tokens & semantic:
            score += 0.2
            reasons.append("The column name resembles a common analytical outcome.")
        if match:
            score += 0.4
            reasons.append("The objective explicitly refers to this column or outcome.")
        if column["missing_pct"] > 30:
            score -= 0.2
            reasons.append("High missingness reduces target suitability.")
        task = "classification" if column["kind"] != "numeric" or (column["unique"] <= 15 and column["unique"] / max(profile["rows"], 1) < 0.1) else "regression"
        score = round(min(1, max(0, score)), 2)
        candidates.append(TargetCandidate(name=name, task=task, score=score, confidence="HIGH" if score >= 0.85 else "MEDIUM" if score >= 0.6 else "LOW", reasons=reasons))
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.name))[:8]


def detect_problem(frame, objective, requested_task="auto", target=None):
    if target is not None and target not in frame.columns:
        raise ValueError("The selected target is not in this dataset.")
    if requested_task == "clustering":
        return {"task": "clustering", "target": None, "reason": "Clustering was selected: discover similar groups without a target."}
    if requested_task != "auto" and not target:
        raise ValueError("Category and numeric prediction need a target column.")
    if not target and not re.search(r"cluster|segment|similar|group", objective, re.I):
        text = objective.lower().replace("_", " ")
        matches = [name for name in frame.columns if len(name) > 2 and name.lower().replace("_", " ") in text]
        if len(matches) == 1:
            target = matches[0]
    if not target:
        return {"task": "clustering", "target": None, "reason": "No unambiguous prediction target was supplied; discover groups instead."}
    task = infer_task(frame[target]) if requested_task == "auto" else requested_task
    reason = ("The target contains categories or a small set of repeated labels." if task == "classification"
              else "The target contains varying numeric values.")
    return {"task": task, "target": target, "reason": reason,
            "target_unique": int(frame[target].nunique()), "target_missing": int(frame[target].isna().sum())}
