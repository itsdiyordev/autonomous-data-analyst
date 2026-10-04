import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from .evidence import record_evidence


def detect_anomalies(frame, columns, evidence):
    columns = [name for name in columns if name in frame and frame[name].replace([np.inf, -np.inf], np.nan).nunique() > 1][:12]
    if not columns or len(frame) < 20:
        return record_evidence(evidence, "anomaly", "Multivariate anomaly review", "Isolation Forest", status="skipped", population_rows=len(frame), limitations=["At least 20 rows and a varying numeric feature are required."])
    sample = frame[columns].replace([np.inf, -np.inf], np.nan).sample(min(10000, len(frame)), random_state=42)
    omitted = [name for name in columns if sample[name].nunique() < 2]
    columns = [name for name in columns if name not in omitted]
    if not columns:
        return record_evidence(evidence, "anomaly", "Multivariate anomaly review", "Isolation Forest", status="skipped", sample_rows=len(sample), population_rows=len(frame), limitations=["The bounded sample contains no varying numeric columns; original rows and columns are retained."])
    sample = sample[columns]
    raw_values = sample.to_numpy(dtype=float)
    normalizers = np.maximum(np.nanmax(np.abs(raw_values), axis=0), 1)
    scaled_values = SimpleImputer(strategy="median").fit_transform(raw_values / normalizers)
    values = scaled_values * normalizers
    model = IsolationForest(n_estimators=80, max_samples=min(256, len(sample)), contamination="auto", random_state=42, n_jobs=1).fit(scaled_values)
    scores = model.decision_function(scaled_values)
    flagged = np.flatnonzero(scores < 0)
    medians = np.median(scaled_values, axis=0)
    scale = np.median(np.abs(scaled_values - medians), axis=0) * 1.4826
    scale = np.where(scale > 0, scale, np.std(scaled_values, axis=0) + 1e-9)
    examples = []
    for index in sorted(flagged, key=lambda value: scores[value])[:12]:
        deviations = np.abs((scaled_values[index] - medians) / scale)
        drivers = np.argsort(deviations)[::-1][:3]
        examples.append({"row_index": int(sample.index[index]), "score": float(scores[index]), "record": sample.iloc[index].to_dict(),
                         "reasons": [{"feature": columns[i], "value": float(values[index, i]), "median": float(medians[i] * normalizers[i]), "robust_deviation": float(deviations[i])} for i in drivers]})
    return record_evidence(evidence, "anomaly", "Multivariate anomaly review", "Isolation Forest", columns=columns,
                           values={"flagged_rows": len(flagged), "flagged_pct": len(flagged) / len(sample) * 100, "threshold": 0, "examples": examples, "omitted_constant_or_empty_sample_columns": omitted, "policy": "Review flags; all original rows are retained. Sample-only unusable dimensions are recorded, not deleted from the dataset."},
                           sample_rows=len(sample), population_rows=len(frame), limitations=["Algorithmic anomaly scores are not error probabilities or proof of invalid data.", "Driver deviations describe unusual values, not causal explanations of the forest score.", "Computed on a deterministic sample when the dataset exceeds 10,000 rows."])
