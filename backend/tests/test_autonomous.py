import io
import json
import zipfile

import numpy as np
import pandas as pd

from app.orchestrator import execute_run
from app.worker import claim_job


def sales_frame():
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.date_range("2022-01-01", periods=48, freq="MS"), 4)
    frame = pd.DataFrame({"order_date": dates, "region": ["North", "South", "East", "West"] * 48, "price": rng.uniform(10, 30, len(dates))})
    frame["sales"] = np.repeat(np.linspace(200, 100, 48), 4) + rng.normal(0, 5, len(frame))
    return frame


def uploaded(client, auth, frame):
    response = client.post("/api/datasets", headers=auth, files={"file": ("analysis.csv", frame.to_csv(index=False).encode())})
    assert response.status_code == 201, response.text
    return response.json()


def completed(client, auth, dataset, objective, **extra):
    response = client.post("/api/analysis-runs", headers=auth, json={"dataset_id": dataset["id"], "objective": objective, "analysis_mode": "autonomous", "budget_seconds": 60, **extra})
    assert response.status_code == 202, response.text
    run = response.json()
    assert claim_job() == run["id"]
    execute_run(run["id"])
    result = client.get(f"/api/analysis-runs/{run['id']}", headers=auth).json()
    assert result["status"] == "completed", result.get("error")
    return result


def test_descriptive_analysis_produces_evidence_report_and_equivalent_rerun(client, auth):
    dataset = uploaded(client, auth, sales_frame())
    run = completed(client, auth, dataset, "Why did sales decrease?")
    analysis = run["result"]["analysis"]
    assert run["task"] == "descriptive" and not analysis["ml_decision"]["use_ml"]
    assert {"association", "group_difference", "temporal", "data_quality", "anomaly"} <= {item["kind"] for item in analysis["evidence"]}
    ids = {item["id"] for item in analysis["evidence"]}
    assert all(set(item["evidence_ids"]) <= ids for item in analysis["insights"] + analysis["recommendations"])
    assert analysis["hypotheses"] and all(hypothesis["evidence_id"] in ids for hypothesis in analysis["hypotheses"])
    assert all(hypothesis["confidence"] in {"HIGH", "MEDIUM", "LOW"} and hypothesis["confidence_reasons"] for hypothesis in analysis["hypotheses"])
    assert all(step["reason"] and step["status"] in {"completed", "skipped"} for step in analysis["plan"]["steps"])
    assert any(event.get("step_id") for event in run["events"])
    assert client.post(f"/api/models/{run['id']}/predict", headers=auth, json={"records": [{}]}).status_code == 409
    report = client.get(f"/api/analysis-runs/{run['id']}/report", headers=auth)
    assert report.status_code == 200 and "Executive Summary" in report.text and "Reproducibility" in report.text
    archive = client.get(f"/api/analysis-runs/{run['id']}/download", headers=auth)
    with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
        assert {"analysis.json", "reproducibility.json", "report.html"} <= set(package.namelist())
        assert "pipeline.joblib" not in package.namelist()
    rerun = client.post(f"/api/analysis-runs/{run['id']}/rerun", headers=auth, json={}).json()
    assert claim_job() == rerun["id"]
    execute_run(rerun["id"])
    repeated = client.get(f"/api/analysis-runs/{rerun['id']}", headers=auth).json()
    assert repeated["status"] == "completed", repeated.get("error")
    assert repeated["result"]["analysis"]["evidence"] == analysis["evidence"]
    assert repeated["result"]["analysis"]["reproducibility"]["configuration_hash"] == analysis["reproducibility"]["configuration_hash"]
    experiment = client.get(f"/api/experiments/{run['config']['experiment_id']}", headers=auth).json()
    assert len(experiment["runs"]) == 2 and experiment["best_run_id"] is None


def test_temporal_forecast_uses_baseline_validation_and_separate_test(client, auth):
    dataset = uploaded(client, auth, sales_frame())
    run = completed(client, auth, dataset, "Forecast future sales for the next month.")
    temporal = next(item for item in run["result"]["analysis"]["evidence"] if item["kind"] == "temporal")
    forecast = temporal["values"]["forecast"]
    assert forecast["status"] == "completed" and len(forecast["horizon"]) == 6
    assert forecast["test_metrics"]["mae"] >= 0
    assert sum(forecast["split"].values()) == 48
    assert "separate" in forecast["evaluation"]


def test_plan_experiment_and_rerun_ownership_and_cancelled_execution(client, auth):
    dataset = uploaded(client, auth, sales_frame())
    plan = client.post("/api/analysis-plans", headers=auth, json={"dataset_id": dataset["id"], "objective": "Understand sales patterns."})
    assert plan.status_code == 200
    run = client.post("/api/analysis-runs", headers=auth, json={"dataset_id": dataset["id"], "objective": "Understand sales patterns.", "analysis_mode": "autonomous"}).json()
    other = client.post("/api/auth/register", json={"email": f"other-{run['id']}@example.com", "name": "Other", "password": "another-password"}).json()
    headers = {"Authorization": f"Bearer {other['token']}"}
    assert client.post("/api/analysis-plans", headers=headers, json={"dataset_id": dataset["id"], "objective": "Understand sales patterns."}).status_code == 404
    assert client.get(f"/api/experiments/{run['config']['experiment_id']}", headers=headers).status_code == 404
    assert client.post(f"/api/analysis-runs/{run['id']}/rerun", headers=headers, json={}).status_code == 404
    assert client.post(f"/api/analysis-runs/{run['id']}/cancel", headers=auth).status_code == 200
    execute_run(run["id"])
    cancelled = client.get(f"/api/analysis-runs/{run['id']}", headers=auth).json()
    assert cancelled["status"] == "cancelled" and cancelled["result"] is None


def test_autonomous_classification_and_evaluated_noncausal_scenario(client, auth):
    dataset = client.post("/api/datasets/demo?kind=churn", headers=auth).json()
    run = completed(client, auth, dataset, "Predict customer churn.")
    result = run["result"]
    assert run["task"] == "classification"
    assert result["calibration"]["brier_score"] >= 0 and result["model_stability"]["status"] in {"stable", "variable", "unstable"}
    assert result["threshold_analysis"]["data"].startswith("separate validation")
    record = {item["name"]: item["example"] for item in result["input_schema"]}
    response = client.post(f"/api/models/{run['id']}/scenario", headers=auth, json={"record": record, "changes": {"monthly_charges": 95}})
    assert response.status_code == 200, response.text
    assert response.json()["label"] == "MODEL-BASED SCENARIO" and "does not establish causal impact" in response.json()["limitation"]
    assert response.json()["evaluation_context"]["test_metrics"] == result["metrics"]
    assert client.post(f"/api/models/{run['id']}/scenario", json={"record": record, "changes": {"monthly_charges": 95}}).status_code == 401
    assert client.post(f"/api/models/{run['id']}/scenario", headers=auth, json={"record": record, "changes": {"churn": "Yes"}}).status_code == 422
    assert json.dumps(result["analysis"]["reproducibility"]).find(dataset["fingerprint"]) >= 0


def test_severe_binary_and_multiclass_ml_keep_appropriate_selection_metrics(client, auth):
    rng = np.random.default_rng(42)
    for labels, expected in [(["Majority"] * 380 + ["A-rare-positive"] * 20, "pr_auc"), (["A", "B", "C"] * 100, "f1_macro")]:
        numeric = np.array([list(dict.fromkeys(labels)).index(label) for label in labels])
        frame = pd.DataFrame({"measurement": numeric * 4 + rng.normal(0, 1, len(labels)), "secondary": rng.normal(size=len(labels)), "target": labels})
        dataset = uploaded(client, auth, frame)
        run = completed(client, auth, dataset, "Predict target categories.", target="target", task="classification")
        result = run["result"]
        assert result["primary_metric"] == expected and "balanced_accuracy" in result["metrics"]
        if expected == "pr_auc":
            assert result["positive_class"] == "A-rare-positive" and result["class_imbalance"]["severe"]
        else:
            assert result["calibration"]["status"] == "evaluated_multiclass"
        assert any(item["family"] == "baseline" and item["status"] == "completed" for item in result["experiments"])
