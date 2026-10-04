import asyncio
import contextlib
import logging
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import current_user, make_token, password_hasher
from .config import settings
from .data import demo_frame, parse_upload, profile_frame, register_frame, safe_json
from .db import AnalysisRun, Dataset, User, get_db, init_db, now
from .llm import answer_dataset
from .schemas import ChatInput, LoginInput, PredictInput, RegisterInput, RunInput
from .static import SPAStaticFiles
from .problem import detect_problem
from .worker import dispatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    init_db()
    task = asyncio.create_task(dispatcher()) if settings.worker_mode == "embedded" else None
    yield
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Analytiq API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_credentials=False,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
rate_buckets = defaultdict(deque)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    started = time.perf_counter()
    request_id = secrets.token_hex(8)
    # Bound authentication attempts. An external rate limiter can additionally protect multi-instance deployments.
    if request.url.path.startswith("/api/auth/") and request.method == "POST":
        key = request.client.host if request.client else "local"
        bucket = rate_buckets[key]
        current = time.monotonic()
        while bucket and current - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= 30:
            return JSONResponse({"detail": "Too many attempts. Try again in a minute."}, status_code=429)
        bucket.append(current)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["Server-Timing"] = f"api;dur={(time.perf_counter() - started) * 1000:.1f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def user_view(user):
    return {"id": user.id, "name": user.name, "email": user.email}


def dataset_view(dataset, detailed=False):
    value = {"id": dataset.id, "name": dataset.name, "filename": dataset.filename,
             "row_count": dataset.row_count, "column_count": dataset.column_count,
             "size_bytes": dataset.size_bytes, "fingerprint": dataset.fingerprint,
             "quality_score": dataset.profile["quality_score"], "created_at": dataset.created_at.isoformat()}
    if detailed:
        value["profile"] = dataset.profile
    return value


def run_view(run):
    return {"id": run.id, "dataset_id": run.dataset_id, "objective": run.objective, "target": run.target,
            "task": run.task, "status": run.status, "progress": run.progress, "stage": run.stage,
            "config": run.config, "events": run.events, "result": run.result, "error": run.error,
            "created_at": run.created_at.isoformat(), "finished_at": run.finished_at.isoformat() if run.finished_at else None}


def owned_dataset(db, identifier, user):
    dataset = db.get(Dataset, identifier)
    if not dataset or dataset.owner_id != user.id:
        raise HTTPException(404, "Dataset not found.")
    return dataset


def owned_run(db, identifier, user):
    run = db.get(AnalysisRun, identifier)
    if not run or run.owner_id != user.id:
        raise HTTPException(404, "Analysis not found.")
    return run


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok", "version": "1.0.0", "llm_enabled": bool(settings.openai_api_key), "demo_enabled": settings.enable_demo}


@app.post("/api/auth/register", status_code=201)
def register(body: RegisterInput, db: Session = Depends(get_db)):
    user = User(email=body.email, name=body.name.strip(), password_hash=password_hasher.hash(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "An account with this email already exists.") from None
    return {"token": make_token(user), "user": user_view(user)}


@app.post("/api/auth/login")
def login(body: LoginInput, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    if not user or not password_hasher.verify(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password.")
    return {"token": make_token(user), "user": user_view(user)}


@app.post("/api/auth/demo")
def demo_login(db: Session = Depends(get_db)):
    if not settings.enable_demo:
        raise HTTPException(403, "Demo workspaces are disabled.")
    user = User(email=f"demo-{secrets.token_hex(8)}@analytiq.local", name="Demo Explorer", password_hash=password_hasher.hash(secrets.token_urlsafe(32)))
    db.add(user)
    db.commit()
    for kind in ("churn", "sales"):
        df, filename = demo_frame(kind)
        register_frame(db, user.id, df, filename)
    return {"token": make_token(user), "user": user_view(user)}


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return user_view(user)


@app.get("/api/dashboard")
def dashboard(user: User = Depends(current_user), db: Session = Depends(get_db)):
    datasets = db.scalars(select(Dataset).where(Dataset.owner_id == user.id).order_by(Dataset.created_at.desc())).all()
    runs = db.scalars(select(AnalysisRun).where(AnalysisRun.owner_id == user.id).order_by(AnalysisRun.created_at.desc()).limit(50)).all()
    return {"datasets": [dataset_view(d) for d in datasets], "runs": [run_view(r) for r in runs],
            "stats": {"datasets": len(datasets), "rows": sum(d.row_count for d in datasets),
                      "models": db.scalar(select(func.count()).select_from(AnalysisRun).where(AnalysisRun.owner_id == user.id, AnalysisRun.status == "completed")),
                      "active_runs": sum(r.status in ("queued", "running") for r in runs),
                      "average_quality": round(sum(d.profile["quality_score"] for d in datasets) / max(len(datasets), 1), 1)}}


@app.get("/api/datasets")
def datasets(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [dataset_view(d) for d in db.scalars(select(Dataset).where(Dataset.owner_id == user.id).order_by(Dataset.created_at.desc())).all()]


@app.post("/api/datasets", status_code=201)
def upload_dataset(file: UploadFile, user: User = Depends(current_user), db: Session = Depends(get_db)):
    # Read at most limit + 1 bytes, including uploads with no Content-Length.
    raw = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Maximum upload size is {settings.max_upload_mb} MB.")
    filename = Path(file.filename or "dataset.csv").name
    dataset = register_frame(db, user.id, parse_upload(raw, filename), filename, raw)
    return dataset_view(dataset, True)


@app.post("/api/datasets/demo", status_code=201)
def add_demo(kind: str = Query("churn", pattern="^(churn|sales)$"), user: User = Depends(current_user), db: Session = Depends(get_db)):
    df, filename = demo_frame(kind)
    return dataset_view(register_frame(db, user.id, df, filename), True)


@app.get("/api/datasets/{dataset_id}")
def dataset_detail(dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    dataset = owned_dataset(db, dataset_id, user)
    if dataset.profile.get("profile_version", 1) < 2:
        dataset.profile = profile_frame(pd.read_parquet(dataset.path))
        db.commit()
    return dataset_view(dataset, True)


@app.get("/api/datasets/{dataset_id}/preview")
def preview(dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100), user: User = Depends(current_user), db: Session = Depends(get_db)):
    dataset = owned_dataset(db, dataset_id, user)
    df = pd.read_parquet(dataset.path).iloc[offset:offset + limit]
    records = df.astype(object).where(pd.notna(df), None).to_dict("records")
    return {"columns": list(df.columns), "records": safe_json(records), "total": dataset.row_count, "offset": offset}


@app.get("/api/datasets/{dataset_id}/scatter")
def scatter(dataset_id: str, x: str, y: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    dataset = owned_dataset(db, dataset_id, user)
    names = [c["name"] for c in dataset.profile["columns"] if c["kind"] == "numeric"]
    if x not in names or y not in names or x == y:
        raise HTTPException(422, "Choose two distinct numeric columns.")
    df = pd.read_parquet(dataset.path, columns=[x, y]).dropna()
    sample = df.sample(min(len(df), 600), random_state=42).rename(columns={x: "x", y: "y"})
    return safe_json({"points": sample.to_dict("records"), "x": x, "y": y, "sampled_rows": len(sample)})


@app.post("/api/datasets/{dataset_id}/chat")
def chat(dataset_id: str, body: ChatInput, user: User = Depends(current_user), db: Session = Depends(get_db)):
    dataset = owned_dataset(db, dataset_id, user)
    return answer_dataset(body.question, dataset.profile)


@app.delete("/api/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    dataset = owned_dataset(db, dataset_id, user)
    if db.scalar(select(func.count()).select_from(AnalysisRun).where(AnalysisRun.dataset_id == dataset_id)):
        raise HTTPException(409, "This dataset is referenced by analysis runs and cannot be deleted.")
    path = Path(dataset.path)
    db.delete(dataset)
    db.commit()
    path.unlink(missing_ok=True)


@app.post("/api/analysis-runs", status_code=202)
def create_run(body: RunInput, user: User = Depends(current_user), db: Session = Depends(get_db)):
    dataset = owned_dataset(db, body.dataset_id, user)
    columns = {c["name"]: c for c in dataset.profile["columns"]}
    try:
        problem = detect_problem(pd.read_parquet(dataset.path), body.objective, body.task, body.target)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.split_strategy != "random" and (body.split_column not in columns or body.split_column == problem["target"]):
        raise HTTPException(422, "Choose a valid split column that differs from the target.")
    active = db.scalar(select(func.count()).select_from(AnalysisRun).where(AnalysisRun.owner_id == user.id, AnalysisRun.status.in_(["queued", "running"])))
    if active >= 3:
        raise HTTPException(429, "You already have three active analyses. Wait for one to finish.")
    task = problem["task"]
    config = body.model_dump(exclude={"dataset_id", "objective", "target", "task"})
    config["problem_detection"] = problem
    run = AnalysisRun(owner_id=user.id, dataset_id=dataset.id, objective=body.objective, target=problem["target"] or "",
                      task=task, config=config,
                      events=[{"at": now().isoformat(), "stage": "Problem detector", "detail": problem["reason"]}])
    db.add(run)
    db.commit()
    return run_view(run)


@app.get("/api/analysis-runs")
def list_runs(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [run_view(r) for r in db.scalars(select(AnalysisRun).where(AnalysisRun.owner_id == user.id).order_by(AnalysisRun.created_at.desc()).limit(100)).all()]


@app.get("/api/analysis-runs/{run_id}")
def run_detail(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return run_view(owned_run(db, run_id, user))


@app.post("/api/analysis-runs/{run_id}/cancel")
def cancel_run(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = owned_run(db, run_id, user)
    if run.status not in ("queued", "running"):
        raise HTTPException(409, "Only queued or running analyses can be cancelled.")
    run.status = "cancelled"
    run.stage = "Cancelled"
    run.finished_at = now()
    db.commit()
    return run_view(run)


def completed_model(db, run_id, user):
    run = owned_run(db, run_id, user)
    if run.status != "completed" or not run.artifact_path:
        raise HTTPException(409, "The model is not ready yet.")
    return run


@lru_cache(maxsize=8)
def load_model(path: str):
    return joblib.load(Path(path) / "pipeline.joblib")


@app.post("/api/models/{run_id}/predict")
def predict(run_id: str, body: PredictInput, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = completed_model(db, run_id, user)
    package = load_model(run.artifact_path)
    for index, record in enumerate(body.records):
        missing = set(package["features"]) - set(record)
        if missing:
            raise HTTPException(422, f"Record {index + 1} is missing: {', '.join(sorted(missing))}")
    frame = pd.DataFrame(body.records)[package["features"]]
    for col in package["numeric"]:
        converted = pd.to_numeric(frame[col], errors="coerce")
        invalid = (frame[col].notna() & converted.isna()) | converted.isin([np.inf, -np.inf])
        if invalid.any():
            raise HTTPException(422, f"'{col}' must contain finite numeric values or null.")
        frame[col] = converted
    for col in package["categorical"]:
        frame[col] = frame[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    raw_prediction = package["pipeline"].predict(frame)
    predictions = []
    if package["task"] == "clustering":
        predictions = [{"cluster": int(value), "label": f"Group {int(value) + 1}"} for value in raw_prediction]
    elif package["encoder"] is not None:
        labels = package["encoder"].inverse_transform(raw_prediction.astype(int))
        probabilities = package["pipeline"].predict_proba(frame)
        classes = package["encoder"].classes_
        for label, row in zip(labels, probabilities):
            predictions.append({"label": label, "probabilities": {str(c): float(p) for c, p in zip(classes, row)}})
    else:
        predictions = [{"value": float(value)} for value in raw_prediction]
    return safe_json({"model_id": run.id, "model_version": package.get("version", "1.0"), "task": run.task, "predictions": predictions})


@app.get("/api/models/{run_id}/download")
def download(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = completed_model(db, run_id, user)
    return FileResponse(Path(run.artifact_path) / "solution.zip", filename=f"analytiq-solution-{run_id[:8]}.zip", media_type="application/zip")


@app.get("/api/analysis-runs/{run_id}/report")
def report_download(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = completed_model(db, run_id, user)
    return FileResponse(Path(run.artifact_path) / "report.html", filename=f"analysis-report-{run_id[:8]}.html", media_type="text/html")


# Registered after API routes so the production dashboard cannot shadow endpoints.
if settings.static_dir:
    if not (settings.static_dir / "index.html").is_file():
        raise RuntimeError(f"The frontend build is missing from STATIC_DIR={settings.static_dir}")
    app.mount("/", SPAStaticFiles(directory=settings.static_dir, html=True), name="dashboard")
