# Analytiq — Autonomous Data Analyst

A working full-stack analytics application: upload tabular data, explore interactive visualizations, launch an autonomous ML workflow, compare models, explain predictions, and export a runnable solution.

**Frontend:** React 19 + TypeScript + Vite + TanStack Query + Recharts.  
**Backend:** FastAPI + SQLAlchemy + pandas + scikit-learn + isolated background worker processes.  
**Storage:** SQLite for local development and the single Docker container; PostgreSQL is available in the multi-service deployment.

## Start here

### Windows — one command

Install **Node.js 22.12+** (24 LTS recommended) and **uv**:

```powershell
winget install --id astral-sh.uv -e
```

Reopen PowerShell, open the project directory, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Open **http://localhost:5173** and select **Try the live demo**. Each demo session receives an isolated workspace with customer churn and retail sales datasets. You can also register a persistent account and upload your own files.

### Manual development setup — Windows, Linux, macOS

In a terminal with `backend` as the working directory:

```bash
uv sync --python 3.12
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal with `frontend` as the working directory:

```bash
npm install
npm run dev
```

- Application: http://localhost:5173
- Interactive API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

Alternatively, run `python scripts/dev.py` from the project root after installing uv and Node.js.

### WSL / Windows-mounted folders

Some Windows-mounted filesystems reject Unix permission changes and package-manager file copies. Store the Python environment in the Linux filesystem if `uv sync` reports `Operation not permitted`:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/analytiq-venv"
uv sync --python 3.12
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The variable must remain set for both commands. The application handles unsupported Unix permissions on its local secret file. For problematic Vite cache directories, set `VITE_CACHE_DIR` to a Linux path before `npm run dev`.

## Features

### Data workspace

- Registration and login with Argon2 password hashes and signed 24-hour JWTs.
- Per-account dataset and model access.
- CSV and XLSX/XLSM ingestion; up to 50 MB, 100,000 rows, and 200 columns.
- Original-value preservation in Parquet, SHA-256 dataset identity, and cached profiling.
- Paginated data preview, column inspection, missing values, duplicate records, IQR outliers, and quality insights.
- A transparent quality score: `100 − missing-cell percentage − duplicate-row percentage`.
- Searchable dataset library and analysis history.

### Visualization

- Selectable numeric histograms and categorical frequency charts.
- Pearson correlation heatmap with hover evidence.
- Interactive scatter plots with selectable axes.
- Model performance cards and cross-validation comparison tables, with metric explanations.
- ROC curves, classification confusion matrices, regression diagnostics, and residual plots.
- Held-out permutation-importance charts, SHAP overview/local-contribution charts, and error examples.
- Colored cluster projections and group profiles.
- Responsive layouts, accessible dialogs, keyboard workspace search, and persistent light/dark themes.

### Autonomous ML

Working engines support **classification, regression, and clustering** on tabular data. The interface guides users through choosing a dataset, defining a goal, and reviewing the training setup.

1. **Data profiler:** column types, missing values, duplicates, IQR outliers, correlations, and distributions.
2. **Problem detector:** infer classification/regression from a target, or clustering without an unambiguous target.
3. **Preprocessing engine:** imputation, missing indicators, encoding, scaling, and calendar/cyclic date features.
4. **Model selector:** baseline, linear, decision-tree, ensemble, and neural-network candidates for supervised tasks; K-Means and Gaussian mixtures for clustering.
5. **Training engine:** fit complete candidate pipelines within the search budget.
6. **Validation / cross-validation:** 2–5 leakage-safe folds using stratified, random, group-aware, or time-aware strategies. Preprocessing is fitted separately inside every fold.
7. **Model comparison:** select by cross-validation mean PR-AUC, macro F1, RMSE, or silhouette. When folds are not feasible for a rare-class group/time split, the recorded separate-validation fallback is used.
8. **Best model:** refit the selected pipeline on train + validation rows.
9. **Prediction:** evaluate once on unseen test data and serve new records through the prediction API.
10. **Explainability:** permutation importance, Tree/Linear/Permutation SHAP, prediction errors, cluster assignment ambiguity, and a grounded explanation.
11. **Final analysis report:** a natural-language summary, comparison evidence, detailed HTML report, and portable model package.

Model-search budgets are checked between candidates. A candidate already running, final refitting, evaluation, and packaging can extend total execution beyond that budget. Long jobs do not run in API request handlers.

Classification requires 2–20 classes with at least 8 examples per class. Training requires at least 40 distinct usable rows and supports up to 100 raw features. Clustering does not require a target; its group-quality scores are unsupervised measures rather than classification accuracy. Potential outliers are flagged and retained. SHAP uses a bounded test/background sample, with its method and sample sizes recorded.

### Deployable outputs

- Single-record and batch predictions, with input-schema validation.
- Classification labels with class probabilities, numeric regression predictions, and new-row cluster assignments.
- Model registry containing configuration, metrics, dataset hash, random seed, runtime versions, and workflow events.
- Downloadable HTML report.
- Downloadable ZIP containing `pipeline.joblib`, `input_schema.json`, `metrics.json`, `pipeline.json`, `predict.py`, portable `app/features.py`, pinned dependencies, README, and Dockerfile.
- Standalone inference package tested independently of the application.

## Optional language-model integration

The application works **without an API key**. Data profiling, model training, prediction, reports, and supported dataset questions run from computed evidence.

Copy `.env.example` to `.env` in the root and set:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
```

Restart the backend. Dataset chat and report writing then use an LLM grounded in computed evidence. An LLM failure falls back to the local evidence-based implementation. The LLM does not calculate model metrics, execute arbitrary code, or replace model evaluation. The ML plan is controlled by typed configuration and deterministic workflow modules.

## Agentation visual feedback

Agentation is installed and connected to the **development frontend**. Its toolbar lets you select any interface element, write a design note, and copy structured feedback.

To preview the frontend against your running Docker backend, run from `frontend` in PowerShell:

```powershell
$env:API_PROXY_TARGET = "http://localhost:8080"
npm run dev
```

Open **http://localhost:5173**, click **Start feedback mode** at the bottom right, select an element, and add your note. Copy the feedback to share the exact element and context. The production Docker dashboard serves the compiled application; the annotation toolbar is a development tool.

## Docker deployment

The root `Dockerfile` packages the **entire working application into one container**:

- Production React dashboard served directly by FastAPI.
- Authenticated API and prediction endpoints.
- Embedded job dispatcher with isolated ML subprocesses.
- SQLite database, uploaded datasets, JWT secret, reports, and trained models stored in one persistent volume.
- A built-in health check and tini for process/signal handling.

Start Docker Desktop in Linux-container mode. From the project root, run:

```bash
docker build -t analytiq:latest .
docker run -d --name analytiq -p 8080:8080 -v analytiq-data:/app/data --restart unless-stopped analytiq:latest
```

Open **http://localhost:8080** and choose **Try the live demo**. API documentation is at **http://localhost:8080/docs**. Node.js and Python are included in the build process; you do not need to install them on the host for Docker deployment.

The default Compose file starts this same single-container setup:

```bash
docker compose up --build -d
```

### Container management

```bash
docker logs -f analytiq
docker inspect --format='{{.State.Health.Status}}' analytiq
docker stop analytiq
docker start analytiq
```

The `analytiq-data` volume preserves your account, uploads, and models across container replacements. Container builds exclude local datasets, existing model files, `.env` secrets, virtual environments, and `node_modules`.

For optional LLM integration with `docker run`, add `-e OPENAI_API_KEY=your-key` before the image name. With Compose, set `OPENAI_API_KEY` in the root `.env`. `MAX_WORKERS` defaults to 2; set it to 1 for a smaller machine. The JWT signing secret is generated and persisted automatically unless `JWT_SECRET` is supplied.

The original PostgreSQL/API/worker/Nginx deployment is preserved in `docker-compose.multi.yml` and can be started with `docker compose -f docker-compose.multi.yml up --build -d`. It uses separate services and is intended for scaling. The default `docker-compose.yml` uses just one container. Run one embedded API instance per SQLite volume.

See [single-container deployment details](docs/DOCKER.md) for rebuilding, moving the image, and persistent storage.

## Tests and build

From `backend`:

```bash
uv run python -m pytest -q
uv run ruff check app tests
```

From `frontend`:

```bash
npm run build
npx playwright install chromium
npm run test:e2e
```

The browser tests use the running development frontend and backend by default. Set `PLAYWRIGHT_BASE_URL=http://localhost:8080` to test the single-container deployment instead. On Linux, Playwright may need browser system libraries (`npx playwright install --with-deps chromium`).

Twelve backend tests cover file ingestion, account isolation, classification/regression/clustering, model families, cross-validation, SHAP, error analysis, outliers, date features, unseen categories, missing input values, invalid prediction inputs, cancellation, group/time separation, portable exported inference, and production static serving. Three browser scenarios exercise the desktop, mobile, and clustering workflows. See [the verification record](docs/VERIFICATION.md) for results, including Agentation feedback checks.

## Project structure

```text
backend/
  app/
    main.py        # REST API, ownership, requests, model serving
    auth.py        # JWT sessions and password hashing
    config.py      # Environment-based settings
    db.py          # SQLAlchemy entities and durable job state
    data.py        # Ingestion, profiling, demo datasets
    ml.py          # Full autonomous ML pipeline, cross-validation, clustering
    features.py    # Portable date/categorical feature engineering
    explain.py     # SHAP and error/assignment analysis
    artifacts.py   # Detailed report and standalone solution export
    problem.py     # Regression/classification/clustering detection
    llm.py         # Optional grounded language-model integration
    worker.py      # Atomic job claiming and isolated execution
    schemas.py    # Validated workflow and inference inputs
  tests/
frontend/
  src/
    pages/         # Overview, datasets, explorer, runs, model studio, settings
    components/    # Charts, dialogs, upload and analysis workflows
  e2e/
docs/ARCHITECTURE.md
docs/DOCKER.md
Dockerfile
docker-compose.yml
docker-compose.multi.yml
start.ps1
scripts/dev.py
```

See [the architecture document](docs/ARCHITECTURE.md) for execution flow, persistence, and the API contract.
