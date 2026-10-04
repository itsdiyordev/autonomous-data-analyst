import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .data import safe_json

logger = logging.getLogger(__name__)


def original_feature(name, features):
    name = name.split("__", 1)[-1]
    name = name.removeprefix("missingindicator_")
    return next((feature for feature in sorted(features, key=len, reverse=True)
                 if name == feature or name.startswith(feature + "_") or name.startswith(feature + "__")), name)


def shap_explanation(pipeline, X_background, X_test, task, features):
    import shap

    preprocess = pipeline[:-1]
    estimator = pipeline[-1]
    background = np.asarray(preprocess.transform(X_background.sample(min(24, len(X_background)), random_state=42)), dtype=float)
    foreground_frame = X_test.sample(min(12, len(X_test)), random_state=42)
    foreground = np.asarray(preprocess.transform(foreground_frame), dtype=float)
    names = preprocess.get_feature_names_out().tolist()
    output = "raw model score" if task == "classification" else "predicted target value"
    try:
        tree_models = (DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier,
                       RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor)
        if isinstance(estimator, tree_models):
            explainer = shap.TreeExplainer(estimator, background)
            values = explainer(foreground, check_additivity=False)
            method = "TreeSHAP"
        elif isinstance(estimator, (Ridge, LogisticRegression)):
            values = shap.LinearExplainer(estimator, background)(foreground)
            method = "LinearSHAP"
        else:
            # Keep the model-agnostic explanation workload bounded while retaining every feature.
            background = background[:8]
            foreground = foreground[:6]
            foreground_frame = foreground_frame.iloc[:6]
            if task == "clustering":
                if hasattr(estimator, "transform"):
                    def predict(matrix):
                        return -estimator.transform(matrix)
                    output = "negative distance to each cluster center"
                else:
                    predict = estimator.predict_proba
                    output = "cluster membership probabilities"
            else:
                predict = estimator.predict_proba if task == "classification" else estimator.predict
                output = "class probabilities" if task == "classification" else "predicted target value"
            explainer = shap.PermutationExplainer(predict, background, seed=42)
            values = explainer(foreground, max_evals=2 * foreground.shape[1] + 1, batch_size=32)
            method = "PermutationSHAP"
        array = np.asarray(values.values)
        magnitude = np.abs(array).mean(axis=0)
        if magnitude.ndim > 1:
            magnitude = magnitude.mean(axis=-1)
        totals = {}
        for name, score in zip(names, magnitude):
            feature = original_feature(name, features)
            totals[feature] = totals.get(feature, 0.0) + float(score)
        importance = sorted([{"feature": feature, "importance": score} for feature, score in totals.items()], key=lambda item: item["importance"], reverse=True)
        directions = []
        if task == "regression" or task == "classification" and (array.ndim == 2 or array.shape[-1] == 2):
            directed = array[:, :, 1] if array.ndim == 3 else array
            for feature in features:
                raw_values = pd.to_numeric(foreground_frame[feature], errors="coerce").to_numpy(dtype=float)
                matching = [index for index, name in enumerate(names) if original_feature(name, features) == feature]
                if not matching:
                    continue
                contribution = directed[:, matching].sum(axis=1)
                usable = np.isfinite(raw_values) & np.isfinite(contribution)
                if usable.sum() < 8 or np.std(raw_values[usable]) == 0 or np.std(contribution[usable]) == 0:
                    continue
                coefficient = float(spearmanr(raw_values[usable], contribution[usable]).statistic)
                if abs(coefficient) >= 0.5:
                    directions.append({"feature": feature, "direction": "higher values associated with higher explained output" if coefficient > 0 else "higher values associated with lower explained output", "coefficient": coefficient,
                                       "sample_rows": int(usable.sum()), "output": "positive-class output" if task == "classification" else "predicted target", "evidence": "Spearman association between observed numeric values and their sampled SHAP contributions", "limitation": "Descriptive small-sample direction, not a general monotonic guarantee or causal effect."})
        predicted = pipeline.predict(foreground_frame)
        examples = []
        for index, row in enumerate(array):
            if row.ndim > 1:
                output_index = min(int(predicted[index]), row.shape[-1] - 1) if task != "regression" else 0
                row = row[:, output_index]
            contributions = {}
            for name, score in zip(names, row):
                feature = original_feature(name, features)
                contributions[feature] = contributions.get(feature, 0.0) + float(score)
            examples.append({"row_index": int(foreground_frame.index[index]), "prediction": safe_json(predicted[index]),
                             "contributions": sorted([{"feature": feature, "value": score} for feature, score in contributions.items()], key=lambda item: abs(item["value"]), reverse=True)})
        return safe_json({"status": "completed", "method": method, "output": output,
                           "sample_rows": len(foreground), "background_rows": len(background),
                           "feature_importance": importance, "examples": examples, "directions": directions})
    except Exception as exc:
        logger.exception("SHAP explanation failed")
        return {"status": "failed", "reason": str(exc)[:250], "feature_importance": [], "examples": []}


def error_analysis(pipeline, X, y, prediction, task, classes):
    if task == "regression":
        residual = np.asarray(y) - prediction
        indexes = np.argsort(np.abs(residual))[::-1][:12]
        return safe_json({"kind": "regression", "mean_residual": float(np.mean(residual)),
                          "p95_absolute_error": float(np.percentile(np.abs(residual), 95)),
                          "examples": [{"row_index": int(X.index[i]), "actual": float(y.iloc[i]), "predicted": float(prediction[i]),
                                        "error": float(abs(residual[i])), "record": X.iloc[i].to_dict()} for i in indexes]})
    if task == "classification":
        probabilities = pipeline.predict_proba(X)
        wrong = np.flatnonzero(np.asarray(y) != prediction)
        indexes = sorted(wrong, key=lambda i: float(probabilities[i].max()), reverse=True)[:12]
        return safe_json({"kind": "classification", "error_count": len(wrong),
                          "class_metrics": classification_report(y, prediction, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
                          "examples": [{"row_index": int(X.index[i]), "actual": classes[int(y.iloc[i])], "predicted": classes[int(prediction[i])],
                                        "error": float(probabilities[i].max()), "record": X.iloc[i].to_dict()} for i in indexes]})
    estimator = pipeline[-1]
    transformed = pipeline[:-1].transform(X)
    if hasattr(estimator, "transform"):
        distances = np.sort(estimator.transform(transformed), axis=1)
        ambiguity = 1 / (1 + distances[:, 1] - distances[:, 0])
    else:
        ambiguity = 1 - estimator.predict_proba(transformed).max(axis=1)
    indexes = np.argsort(ambiguity)[::-1][:12]
    return safe_json({"kind": "cluster_ambiguity", "note": "There are no ground-truth labels. These are ambiguous group assignments, not measured prediction errors.",
                      "examples": [{"row_index": int(X.index[i]), "predicted": f"Group {int(prediction[i]) + 1}", "error": float(ambiguity[i]),
                                    "record": X.iloc[i].to_dict()} for i in indexes]})
