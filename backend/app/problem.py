import re

from .data import infer_task


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
