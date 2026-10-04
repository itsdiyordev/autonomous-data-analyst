"""Task 1: externally observable security, concurrency and evaluation regressions."""
import asyncio
import json
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import timedelta

import jwt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.exc import OperationalError

from app import main, ml, orchestrator, worker
from app.config import settings
from app.data import profile_frame
from app.db import AnalysisRun, SessionLocal, User, now
from app.features import FeatureEngineer
from app.planner import create_plan
from app.schemas import PredictInput, ScenarioInput


@pytest.fixture
def dataset(client, auth):
    frame = pd.DataFrame({"measurement": np.arange(64), "target": np.arange(64) * 2.3})
    response = client.post("/api/datasets", headers=auth, files={"file": ("baseline.csv", frame.to_csv(index=False).encode())})
    assert response.status_code == 201
    value = response.json()
    yield value
    # These tests exercise queuing/state without training. Do not leave jobs for
    # the independently preserved pipeline tests' global worker-claim checks.
    with SessionLocal() as db:
        db.execute(update(AnalysisRun).where(AnalysisRun.dataset_id == value["id"], AnalysisRun.status.in_(["queued", "running"])).values(status="cancelled"))
        db.commit()


def queued(client, auth, dataset):
    response = client.post("/api/analysis-runs", headers=auth, json={"dataset_id": dataset["id"], "objective": "Predict target from measurement", "target": "target", "budget_seconds": 30})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_http_errors_are_structured_and_have_security_headers(client):
    response = client.get("/api/datasets")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"
    assert response.json()["error"]["message"]
    assert isinstance(response.json()["error"]["details"], dict)
    assert response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_validation_errors_do_not_echo_passwords(client):
    main.rate_buckets.clear()
    secret = "private-password-" * 20
    response = client.post("/api/auth/register", json={"email": "invalid", "name": "Analyst", "password": secret})
    assert response.status_code == 422
    assert secret not in response.text
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["details"]["issues"]


def test_blank_registered_name_is_rejected(client):
    main.rate_buckets.clear()
    response = client.post("/api/auth/register", json={"email": "blank-name@example.com", "name": "   ", "password": "valid-password"})
    assert response.status_code == 422


def test_existing_blank_names_have_a_safe_display_contract(client, auth):
    identifier = client.get("/api/auth/me", headers=auth).json()["id"]
    with SessionLocal() as db:
        db.get(User, identifier).name = "   "
        db.commit()
    response = client.get("/api/auth/me", headers=auth)
    assert response.status_code == 200 and response.json()["name"] == "Analyst"


def test_unexpected_api_errors_are_sanitized(monkeypatch, auth):
    def broken(*args):
        raise RuntimeError("private /app/data/internal-secret.txt stack information")

    monkeypatch.setattr(main, "parse_upload", broken)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/api/datasets", headers=auth, files={"file": ("safe.csv", b"x\n1\n")})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "internal-secret" not in response.text
    assert response.headers["x-request-id"]


@pytest.mark.parametrize("value", [[1, 2], {"nested": "object"}, float("inf")])
def test_prediction_contract_rejects_non_scalar_or_nonfinite_values(value):
    with pytest.raises(ValidationError):
        PredictInput(records=[{"measurement": value}])
    with pytest.raises(ValidationError):
        ScenarioInput(record={"measurement": 1}, changes={"measurement": value})


@pytest.mark.parametrize("kind", ["missing_expiry", "expired", "tampered"])
def test_session_tokens_require_valid_signature_and_expiry(client, auth, kind):
    original = auth["Authorization"].split(" ", 1)[1]
    payload = jwt.decode(original, options={"verify_signature": False})
    if kind == "missing_expiry":
        payload.pop("exp")
    elif kind == "expired":
        payload["exp"] = now() - timedelta(seconds=1)
    token = jwt.encode(payload, "incorrect-signing-key-" * 3 if kind == "tampered" else settings.jwt_secret, algorithm="HS256")
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


@pytest.mark.parametrize("multipart", [False, True])
def test_streamed_bodies_are_bounded_before_full_parsing(monkeypatch, auth, multipart):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    boundary = "task1-boundary"
    prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.csv"\r\nContent-Type: text/csv\r\n\r\nx\n'.encode() if multipart else b'{"email":"x@example.com","password":"')
    tail = f"\r\n--{boundary}--\r\n".encode() if multipart else b'"}'
    chunks = [prefix, *([b"x" * (512 * 1024)] * 10), tail]
    received, sent = [], []
    headers = [(b"content-type", f"multipart/form-data; boundary={boundary}".encode() if multipart else b"application/json"), (b"transfer-encoding", b"chunked")]
    if multipart:
        headers.append((b"authorization", auth["Authorization"].encode()))
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/datasets" if multipart else "/api/auth/login", "raw_path": b"/api/datasets" if multipart else b"/api/auth/login", "query_string": b"", "root_path": "", "headers": headers, "client": ("stream-test", 123), "server": ("testserver", 80)}

    async def receive():
        index = len(received)
        if index >= len(chunks):
            return {"type": "http.disconnect"}
        received.append(index)
        return {"type": "http.request", "body": chunks[index], "more_body": index < len(chunks) - 1}

    async def send(message):
        sent.append(message)

    asyncio.run(main.app(scope, receive, send))
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    assert start["status"] == 413
    assert json.loads(body)["error"]["code"] == "REQUEST_TOO_LARGE"
    assert len(received) < len(chunks)


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_cancel_cannot_overwrite_a_concurrent_terminal_transition(client, auth, dataset, monkeypatch, terminal):
    identifier = queued(client, auth, dataset)
    original = main.owned_run

    def stale_read(db, run_id, user):
        run = original(db, run_id, user)
        with SessionLocal() as writer:
            writer.execute(update(AnalysisRun).where(AnalysisRun.id == run_id).values(status=terminal, stage="Terminal state"))
            writer.commit()
        return run

    monkeypatch.setattr(main, "owned_run", stale_read)
    assert client.post(f"/api/analysis-runs/{identifier}/cancel", headers=auth).status_code == 409
    with SessionLocal() as db:
        assert db.get(AnalysisRun, identifier).status == terminal


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_workers_do_not_modify_terminal_runs(client, auth, dataset, terminal):
    identifier = queued(client, auth, dataset)
    with SessionLocal() as db:
        db.execute(update(AnalysisRun).where(AnalysisRun.id == identifier).values(status=terminal, stage="Terminal state", progress=17))
        db.commit()
    with pytest.raises(ml.Cancelled):
        ml.update_progress(identifier, 90, "Late work", "Must not be written")
    with pytest.raises(ml.Cancelled):
        orchestrator._save_plan(identifier, create_plan(dataset["profile"], "Understand data patterns"))
    with SessionLocal() as db:
        run = db.get(AnalysisRun, identifier)
        assert (run.status, run.stage, run.progress) == (terminal, "Terminal state", 17)


@pytest.mark.parametrize("autonomous", [False, True])
def test_background_failures_are_structured_and_do_not_disclose_internals(client, auth, dataset, monkeypatch, autonomous):
    identifier = queued(client, auth, dataset)
    with SessionLocal() as db:
        run = db.get(AnalysisRun, identifier)
        run.status = "running"
        run.config = {**run.config, "analysis_mode": "autonomous" if autonomous else "ml"}
        db.commit()

    def broken(*args, **kwargs):
        raise RuntimeError("private /app/data/internal-secret.txt stack information")

    monkeypatch.setattr(orchestrator if autonomous else ml, "_analyze" if autonomous else "_train", broken)
    (orchestrator.execute_run if autonomous else ml.train_run)(identifier)
    result = client.get(f"/api/analysis-runs/{identifier}", headers=auth).json()
    assert result["status"] == "failed"
    assert "internal-secret" not in json.dumps(result)
    assert result["failure"]["code"] == "ANALYSIS_FAILED"
    assert result["error"] == result["failure"]["message"]


def test_a_late_worker_exception_cannot_replace_completed_state(client, auth, dataset, monkeypatch):
    identifier = queued(client, auth, dataset)

    def completed_then_error(*args, **kwargs):
        with SessionLocal() as db:
            db.execute(update(AnalysisRun).where(AnalysisRun.id == identifier).values(status="completed"))
            db.commit()
        raise RuntimeError("late cleanup failure")

    monkeypatch.setattr(ml, "_train", completed_then_error)
    ml.train_run(identifier)
    assert client.get(f"/api/analysis-runs/{identifier}", headers=auth).json()["status"] == "completed"


def test_insufficient_training_data_has_an_actionable_error_code(client, auth):
    response = client.post("/api/datasets", headers=auth, files={"file": ("small.csv", b"measurement,target\n1,2\n2,4\n3,6\n")})
    identifier = queued(client, auth, response.json())
    with SessionLocal() as db:
        db.execute(update(AnalysisRun).where(AnalysisRun.id == identifier).values(status="running"))
        db.commit()
    ml.train_run(identifier)
    run = client.get(f"/api/analysis-runs/{identifier}", headers=auth).json()
    assert run["status"] == "failed" and run["failure"]["code"] == "INSUFFICIENT_DATA"
    assert "40" in run["failure"]["message"]


@pytest.mark.parametrize("failure", ["submit", "database"])
def test_dispatcher_recovers_from_submission_or_transient_database_failure(client, auth, dataset, monkeypatch, failure):
    identifiers = [queued(client, auth, dataset) for _ in range(2 if failure == "submit" else 1)]
    submissions, claims = [], []
    original_claim, original_sleep = worker.claim_job, asyncio.sleep

    class FinishingPool:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, identifier):
            submissions.append(identifier)
            if failure == "submit" and len(submissions) == 1:
                raise BrokenProcessPool("controlled submission failure")
            with SessionLocal() as db:
                db.execute(update(AnalysisRun).where(AnalysisRun.id == identifier).values(status="completed"))
                db.commit()
            future = Future()
            future.set_result(None)
            return future

        def shutdown(self, **kwargs):
            pass

    def claim():
        claims.append(True)
        if failure == "database" and len(claims) == 1:
            raise OperationalError("controlled", {}, RuntimeError("transient database failure"))
        return original_claim()

    async def advance(seconds):
        with SessionLocal() as db:
            if all(db.get(AnalysisRun, identifier).status in {"completed", "failed"} for identifier in identifiers):
                raise asyncio.CancelledError()
        await original_sleep(0)

    monkeypatch.setattr(worker, "ProcessPoolExecutor", FinishingPool)
    monkeypatch.setattr(worker, "claim_job", claim)
    monkeypatch.setattr(worker.asyncio, "sleep", advance)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.dispatcher())
    assert client.get(f"/api/analysis-runs/{identifiers[-1]}", headers=auth).json()["status"] == "completed"
    if failure == "submit":
        first = client.get(f"/api/analysis-runs/{identifiers[0]}", headers=auth).json()
        assert first["status"] == "failed" and first["failure"]["code"] == "WORKER_INTERRUPTED"


def test_concurrent_requests_respect_the_three_active_run_limit(client, auth, dataset):
    def create(_):
        return client.post("/api/analysis-runs", headers=auth, json={"dataset_id": dataset["id"], "objective": "Predict target from measurement", "target": "target"}).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(create, range(8)))
    assert statuses.count(202) == 3
    assert statuses.count(429) == 5


def test_dataset_referenced_by_empty_experiment_cannot_be_deleted(client, auth, dataset):
    assert client.post("/api/experiments", headers=auth, json={"dataset_id": dataset["id"], "name": "Empty experiment"}).status_code == 201
    assert client.delete(f"/api/datasets/{dataset['id']}", headers=auth).status_code == 409
    assert client.get(f"/api/datasets/{dataset['id']}/preview", headers=auth).status_code == 200


def test_expired_rate_buckets_do_not_permanently_exhaust_capacity(client, monkeypatch):
    current = main.time.monotonic()
    main.rate_buckets.clear()
    for index in range(10000):
        main.rate_buckets[(f"expired-{index}", "auth")].append(current - 120)
    try:
        response = client.post("/api/auth/login", json={"email": "unknown@example.com", "password": "incorrect"})
        assert response.status_code == 401
    finally:
        main.rate_buckets.clear()


def test_chronological_splits_and_folds_keep_equal_timestamps_together():
    frame = pd.DataFrame({"date": np.repeat(pd.date_range("2024-01-01", periods=20), 11), "__sort_time": np.arange(220), "target": np.arange(220) * 2.0})
    config = {"split_strategy": "chronological", "split_column": "date", "test_size": 0.23, "cv_folds": 3}
    train, validation, test = ml.split_data(frame, "target", "regression", config)
    assert train.date.max() < validation.date.min() < test.date.min()
    assert all("__sort_time" in part.columns for part in (train, validation, test))
    folds, metadata = ml.cv_plan(train, "target", "regression", config)
    assert folds and metadata["status"] == "completed"
    for fitting, held_out in folds:
        assert train.iloc[fitting].date.max() < train.iloc[held_out].date.min()


def test_date_feature_name_collisions_are_not_silently_overwritten():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10), "date__year": np.arange(10)})
    with pytest.raises(ValueError, match="collid|conflict"):
        FeatureEngineer(("date",)).fit_transform(frame)


@pytest.mark.parametrize("task,target", [("classification", "churn"), ("regression", "sales")])
def test_explicit_supervised_tasks_keep_their_target_when_objective_mentions_segments(task, target):
    frame = pd.DataFrame({"measurement": np.arange(128), "churn": ["No", "Yes"] * 64, "sales": np.arange(128) * 1.5})
    plan = create_plan(profile_frame(frame), f"Predict {target} across customer segments.", target, task)
    assert plan.ml_decision.use_ml and plan.ml_decision.task == task
    assert plan.ml_decision.target == target and plan.intent == "predictive"


def test_explicit_clustering_is_target_free_despite_future_words():
    profile = profile_frame(pd.DataFrame({"measurement": np.arange(64), "constant_target": [1] * 64}))
    plan = create_plan(profile, "Find customer groups for next month.", "constant_target", "clustering")
    assert plan.intent == "segmentation" and plan.ml_decision.use_ml
    assert plan.ml_decision.task == "clustering" and plan.ml_decision.target is None


def test_numeric_classification_labels_use_nominal_statistics_and_not_anomaly_distances(client, auth):
    labels = np.repeat(np.arange(18), 12)
    frame = pd.DataFrame({"measurement": labels * 2 + np.random.default_rng(42).normal(0, 0.2, len(labels)), "target": labels})
    dataset = client.post("/api/datasets", headers=auth, files={"file": ("nominal.csv", frame.to_csv(index=False).encode())}).json()
    response = client.post("/api/analysis-runs", headers=auth, json={"dataset_id": dataset["id"], "objective": "Predict target categories", "target": "target", "task": "classification", "analysis_mode": "autonomous", "budget_seconds": 30})
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    with SessionLocal() as db:
        db.execute(update(AnalysisRun).where(AnalysisRun.id == identifier).values(status="running"))
        db.commit()
    orchestrator.execute_run(identifier)
    run = client.get(f"/api/analysis-runs/{identifier}", headers=auth).json()
    assert run["status"] == "completed", run["error"]
    evidence = run["result"]["analysis"]["evidence"]
    assert any(item["kind"] == "group_difference" and item["columns"][0] == "target" for item in evidence)
    assert not any("coefficient" in item["values"] and "target" in item["columns"] for item in evidence)
    assert all("target" not in item["columns"] for item in evidence if item["kind"] == "anomaly")


def test_uploaded_serialized_models_are_never_deserialized(client, auth, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An uploaded model must never reach joblib.load")

    monkeypatch.setattr(main.joblib, "load", forbidden)
    for name in ("untrusted.pkl", "untrusted.pickle", "untrusted.joblib"):
        assert client.post("/api/datasets", headers=auth, files={"file": (name, b"untrusted serialized bytes")}).status_code == 415
