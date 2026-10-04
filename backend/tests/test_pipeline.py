import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from app.data import profile_frame
from app.ml import cv_plan, train_run
from app.worker import claim_job


def run_uploaded(client, auth, frame, task, target=None):
    upload = client.post("/api/datasets", headers=auth, files={"file": ("pipeline_data.csv", frame.to_csv(index=False).encode())})
    assert upload.status_code == 201, upload.text
    response = client.post("/api/analysis-runs", headers=auth, json={"dataset_id": upload.json()["id"], "task": task, "target": target,
                           "objective": "Find groups of similar records" if target is None else f"Predict {target} from available data", "budget_seconds": 60})
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    assert claim_job() == run_id
    train_run(run_id)
    result = client.get(f"/api/analysis-runs/{run_id}", headers=auth).json()
    assert result["status"] == "completed", result.get("error")
    return result


def test_clustering_detection_validation_explanations_and_prediction(client, auth):
    values, _ = make_blobs(n_samples=180, centers=3, n_features=2, cluster_std=0.35, random_state=7)
    frame = pd.DataFrame(values, columns=["spend", "visits"])
    frame["customer_id"] = [f"C{i}" for i in range(len(frame))]
    run = run_uploaded(client, auth, frame, "auto")
    result = run["result"]
    assert run["task"] == "clustering"
    assert result["target"] is None
    assert result["metrics"]["silhouette"] > 0.5
    assert result["shap"]["status"] == "completed", result["shap"]
    assert result["cross_validation"]["folds"] == 3
    assert result["error_analysis"]["kind"] == "cluster_ambiguity"
    assert len(result["cluster_profiles"]) >= 2
    record = {field["name"]: field["example"] for field in result["input_schema"]}
    response = client.post(f"/api/models/{run['id']}/predict", headers=auth, json={"records": [record]})
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["predictions"][0]["cluster"], int)


def test_date_engineering_is_portable_in_exported_solution(client, auth, tmp_path):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=160)
    frame = pd.DataFrame({"order_date": dates.strftime("%Y-%m-%d"), "quantity": rng.integers(1, 30, len(dates))})
    frame["revenue"] = dates.month * 15 + frame.quantity * 7 + rng.normal(0, 1, len(dates))
    run = run_uploaded(client, auth, frame, "regression", "revenue")
    assert "order_date" in run["result"]["feature_engineering"]["date_columns"]
    assert run["result"]["shap"]["status"] == "completed", run["result"]["shap"]
    archive = client.get(f"/api/models/{run['id']}/download", headers=auth)
    with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
        assert "app/features.py" in package.namelist()
        package.extractall(tmp_path)
    record = {field["name"]: field["example"] for field in run["result"]["input_schema"]}
    record["order_date"] = "2025-06-01"
    input_path = tmp_path / "records.json"
    input_path.write_text(json.dumps([record]))
    offline = subprocess.run([sys.executable, str(tmp_path / "predict.py"), str(input_path)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert offline.returncode == 0, offline.stderr
    api = client.post(f"/api/models/{run['id']}/predict", headers=auth, json={"records": [record]})
    assert api.status_code == 200, api.text
    assert np.isclose(json.loads(offline.stdout)["predictions"][0], api.json()["predictions"][0]["value"])


def test_outlier_profiling_and_group_time_cv_separation():
    profile = profile_frame(pd.DataFrame({"value": list(range(10)) + [1000]}))
    assert profile["outlier_rows"] == 1
    assert profile["columns"][0]["outliers"] == 1
    frame = pd.DataFrame({"group": np.repeat(np.arange(20), 10), "target": np.arange(200, dtype=float)})
    folds, _ = cv_plan(frame, "target", "regression", {"split_strategy": "group", "split_column": "group", "cv_folds": 3})
    for train, validation in folds:
        assert not set(frame.iloc[train].group) & set(frame.iloc[validation].group)
    folds, _ = cv_plan(frame, "target", "regression", {"split_strategy": "chronological", "cv_folds": 3})
    for train, validation in folds:
        assert train.max() < validation.min()
