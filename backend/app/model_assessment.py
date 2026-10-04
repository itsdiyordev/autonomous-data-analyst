"""Model suitability, calibration and stability; selection remains development-only."""
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score


def eligibility(task, family, rows, width, class_counts=None):
    if family == "baseline":
        return True, "A baseline is always eligible to contextualize model quality."
    if family == "neural_network" and (rows < 100 or width > 256 or class_counts is not None and min(class_counts) < 12):
        return False, "Neural training needs at least 100 search rows, at most 256 estimated encoded inputs and adequate examples per class."
    if family == "ensemble" and width > 1200:
        return False, "Dense ensemble training is bounded to 1,200 estimated encoded features to limit memory use."
    if task == "clustering" and width > 800:
        return False, "Distance/density clustering is unsuitable for this high-dimensional dense feature space."
    return True, f"Eligible for {rows} development-search rows and approximately {width} encoded inputs."


def imbalance(frame, target):
    counts = frame[target].dropna().astype(str).value_counts()
    total = int(counts.sum())
    distribution = [{"class": name, "rows": int(count), "share": float(count / max(total, 1))} for name, count in counts.items()]
    share = float(counts.min() / total) if total else 0
    return {"distribution": distribution, "minority_share": share, "severe": share < 0.1,
            "policy": "Binary selection uses positive-outcome PR-AUC; multiclass selection uses macro F1. Balanced accuracy and per-class metrics are reported; suitable models use class weights. No automatic oversampling.",
            "note": "By default the binary positive outcome is the minority class; its label is recorded."}


def stability(cv):
    if cv.get("mean") is None or cv.get("std") is None:
        return {"status": "unavailable", "reason": "Cross-validation was unavailable; separate validation does not estimate fold stability."}
    ratio = abs(cv["std"]) / max(abs(cv["mean"]), 1e-9)
    return {"status": "stable" if ratio <= 0.1 else "variable" if ratio <= 0.2 else "unstable", "relative_std": ratio,
            "rule": "SD / |CV mean| ≤ 0.10: stable; ≤ 0.20: variable; otherwise unstable. This is a heuristic, not a confidence interval."}


def inference_timing(pipeline, frame):
    sample = frame.head(min(32, len(frame)))
    started = time.perf_counter()
    pipeline.predict(sample)
    return {"seconds": time.perf_counter() - started, "rows": len(sample), "note": "Wall-clock batch timing on this machine; not a throughput guarantee."}


def calibration(pipeline, X, y, classes):
    probability = pipeline.predict_proba(X)
    if len(classes) != 2:
        actual = np.eye(len(classes))[np.asarray(y, dtype=int)]
        return {"status": "evaluated_multiclass", "brier_score": float(np.square(probability - actual).sum(axis=1).mean()), "definition": "Multiclass sum-of-squares Brier score; not directly comparable with the binary definition.", "reliability": [], "note": "Raw probabilities are not guaranteed calibrated. No probability recalibration was fitted on test data."}
    positive = probability[:, 1]
    bins = []
    for index in range(10):
        selected = (positive >= index / 10) & (positive <= 1 if index == 9 else positive < (index + 1) / 10)
        if selected.any():
            bins.append({"predicted_probability": float(positive[selected].mean()), "observed_frequency": float(np.asarray(y)[selected].mean()), "rows": int(selected.sum())})
    error = sum(abs(bin["predicted_probability"] - bin["observed_frequency"]) * bin["rows"] for bin in bins) / max(len(X), 1)
    return {"status": "good" if error < 0.05 else "moderate" if error < 0.1 else "poor", "brier_score": float(brier_score_loss(y, positive)), "expected_calibration_error": error,
            "reliability": bins, "positive_class": classes[1], "rule": "Ten equal-width bins; ECE < .05/.10 gives good/moderate descriptive calibration status.",
            "note": "Raw model probabilities are not automatically calibrated. This evaluates test calibration; no calibrator or decision threshold was fitted on test rows."}


def threshold_analysis(pipeline, X, y):
    probabilities = pipeline.predict_proba(X)[:, 1]
    rows = []
    for threshold in (0.1, 0.25, 0.5, 0.75, 0.9):
        prediction = (probabilities >= threshold).astype(int)
        rows.append({"threshold": threshold, "precision": float(precision_score(y, prediction, zero_division=0)), "recall": float(recall_score(y, prediction, zero_division=0)), "f1": float(f1_score(y, prediction, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction))})
    return {"rows": rows, "data": "separate validation before final refit", "policy": "Exploratory threshold tradeoffs; the operational threshold remains the estimator default. Do not optimize it on test data."}
