import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.db import AnalysisRun, SessionLocal
from app.ml import split_data, train_run
from app.worker import claim_job


def completed_run(client, auth, kind, target):
    dataset = client.post(f"/api/datasets/demo?kind={kind}", headers=auth).json()
    response = client.post("/api/analysis-runs", headers=auth, json={"dataset_id": dataset["id"], "target": target, "task": "auto", "objective": f"Predict {target} and explain the factors", "budget_seconds": 30})
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    assert claim_job() == identifier
    train_run(identifier)
    run = client.get(f"/api/analysis-runs/{identifier}", headers=auth).json()
    assert run["status"] == "completed", run.get("error")
    return dataset, run


def test_upload_profile_and_tenant_isolation(client, auth):
    csv = b"customer_id,amount,segment,outcome\nA,10,West,Yes\nB,,East,No\nC,30,West,Yes\n"
    response = client.post("/api/datasets", headers=auth, files={"file": ("customers.csv", csv, "text/csv")})
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["profile"]["missing_cells"] == 1
    assert dataset["row_count"] == 3
    preview = client.get(f"/api/datasets/{dataset['id']}/preview", headers=auth).json()
    assert preview["records"][1]["amount"] is None
    other = client.post("/api/auth/register", json={"email": "other@example.com", "name": "Other Analyst", "password": "another-password"}).json()
    other_auth = {"Authorization": f"Bearer {other['token']}"}
    assert client.get(f"/api/datasets/{dataset['id']}", headers=other_auth).status_code == 404
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 401
    assert client.post("/api/datasets", headers=auth, files={"file": ("script.py", b"print(1)")}).status_code == 415
    answer = client.post(f"/api/datasets/{dataset['id']}/chat", headers=auth, json={"question": "What values are missing?"}).json()
    assert answer["source"] == "computed"
    assert "1 missing" in answer["answer"]


def test_classification_pipeline_predictions_and_export(client, auth, tmp_path):
    dataset, run = completed_run(client, auth, "churn", "churn")
    result = run["result"]
    assert run["task"] == "classification"
    assert result["metrics"]["pr_auc"] > 0.4
    assert result["shap"]["status"] == "completed", result["shap"]
    assert result["cross_validation"]["folds"] == 3
    assert len(result["cross_validation"]["scores"]) == 3
    assert {"linear", "tree", "ensemble", "neural_network"} <= {item["family"] for item in result["experiments"] if item["status"] == "completed"}
    assert len(result["pipeline"]) == 11
    assert result["error_analysis"]["kind"] == "classification"
    assert sum(result["split"][name] for name in ("train", "validation", "test")) == dataset["row_count"]
    assert any(f["feature"] == "customer_id" for f in result["excluded_features"])
    record = {f["name"]: f["example"] for f in result["input_schema"]}
    record["contract_type"] = "Previously unseen contract"
    record["usage_gb"] = None
    response = client.post(f"/api/models/{run['id']}/predict", headers=auth, json={"records": [record]})
    assert response.status_code == 200, response.text
    prediction = response.json()["predictions"][0]
    assert prediction["label"] in result["classes"]
    assert sum(prediction["probabilities"].values()) == pytest.approx(1)
    assert client.post(f"/api/models/{run['id']}/predict", headers=auth, json={"records": [{}]}).status_code == 422
    invalid = {**record, "monthly_charges": "not-a-number"}
    assert client.post(f"/api/models/{run['id']}/predict", headers=auth, json={"records": [invalid]}).status_code == 422
    archive_response = client.get(f"/api/models/{run['id']}/download", headers=auth)
    assert archive_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        assert {"pipeline.joblib", "predict.py", "input_schema.json", "Dockerfile", "report.html"} <= set(archive.namelist())
        archive.extractall(tmp_path)
    input_file = tmp_path / "new-records.json"
    input_file.write_text(json.dumps([record]))
    execution = subprocess.run([sys.executable, str(tmp_path / "predict.py"), str(input_file)], capture_output=True, text=True, timeout=30)
    assert execution.returncode == 0, execution.stderr
    assert json.loads(execution.stdout)["predictions"][0] == prediction["label"]
    assert client.delete(f"/api/datasets/{dataset['id']}", headers=auth).status_code == 409


def test_regression_pipeline_on_new_data(client, auth):
    _, run = completed_run(client, auth, "sales", "monthly_revenue")
    result = run["result"]
    assert run["task"] == "regression"
    assert result["metrics"]["r2"] > 0.75
    assert result["shap"]["status"] == "completed", result["shap"]
    assert result["error_analysis"]["p95_absolute_error"] > 0
    record = {f["name"]: f["example"] for f in result["input_schema"]}
    response = client.post(f"/api/models/{run['id']}/predict", headers=auth, json={"records": [record, record]})
    assert response.status_code == 200, response.text
    assert response.json()["predictions"][0]["value"] == response.json()["predictions"][1]["value"]


def test_cancelled_job_cannot_publish_a_model(client, auth):
    dataset = client.post("/api/datasets/demo", headers=auth).json()
    response = client.post("/api/analysis-runs", headers=auth, json={"dataset_id": dataset["id"], "target": "churn", "objective": "Predict customer churn"})
    identifier = response.json()["id"]
    assert client.post(f"/api/analysis-runs/{identifier}/cancel", headers=auth).json()["status"] == "cancelled"
    train_run(identifier)
    with SessionLocal() as db:
        run = db.get(AnalysisRun, identifier)
        assert run.status == "cancelled"
        assert run.artifact_path is None


def test_group_and_time_splits_do_not_leak():
    frame = pd.DataFrame({"group": np.repeat(np.arange(20), 10), "date": pd.date_range("2024-01-01", periods=200), "target": np.arange(200) * 2.0})
    train, val, test = split_data(frame, "target", "regression", {"split_strategy": "group", "split_column": "group"})
    assert not (set(train.group) & set(val.group) or set(train.group) & set(test.group) or set(val.group) & set(test.group))
    train, val, test = split_data(frame, "target", "regression", {"split_strategy": "chronological", "split_column": "date"})
    assert train.date.max() < val.date.min() < test.date.min()


def test_excel_upload_and_scatter(client, auth):
    content = io.BytesIO()
    pd.DataFrame({"cost": [1, 2, 3], "revenue": [5, 8, 9]}).to_excel(content, index=False)
    response = client.post("/api/datasets", headers=auth, files={"file": ("sales.xlsx", content.getvalue())})
    assert response.status_code == 201, response.text
    dataset = response.json()
    scatter = client.get(f"/api/datasets/{dataset['id']}/scatter?x=cost&y=revenue", headers=auth).json()
    assert len(scatter["points"]) == 3
