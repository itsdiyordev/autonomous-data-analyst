# One-container deployment

## Build and start on Windows

Start Docker Desktop, open PowerShell, and run:

```powershell
cd C:\Users\diyor\autonomous-data-analyst
docker build -t analytiq:latest .
docker run -d --name analytiq -p 8080:8080 -v analytiq-data:/app/data --restart unless-stopped analytiq:latest
```

Open **http://localhost:8080**. Documentation and API are served from the same container:

- `/` — web dashboard
- `/docs` — interactive API documentation
- `/api/health` — application and database health
- `/api/*` — authenticated analytics and prediction endpoints

The initial build downloads Node/Python build images and Python dependencies. Subsequent builds reuse cached layers.

## What is inside the image

```text
Single container: analytiq
├── tini (PID 1; signal forwarding and process reaping)
├── FastAPI / Uvicorn (one API process on port 8080)
│   ├── compiled React dashboard
│   ├── authenticated REST API
│   ├── cached prediction pipelines
│   └── embedded persistent job dispatcher
├── isolated ML subprocesses (MAX_WORKERS, default 2)
└── /app/data → Docker volume analytiq-data
    ├── analytiq.db
    ├── .jwt-secret
    ├── datasets/
    ├── models/
    └── reports/
```

The final image contains a Python runtime, installed backend dependencies, application code, and built frontend assets. Node.js and frontend build tools are used in a build stage and are not included in the runtime image. The application runs as a non-root user.

## One-command startup with Compose

```bash
docker compose up --build -d
```

The default Compose file has **one service** and creates **one application container**. It uses the same image, port, and persistent volume as the `docker run` command. Run one instance at a time against a given SQLite volume.

## Configuration

| Environment variable | Default / purpose |
|---|---|
| `DATA_DIR` | `/app/data` |
| `DATABASE_URL` | `sqlite:////app/data/analytiq.db` |
| `STATIC_DIR` | `/app/frontend/dist` |
| `WORKER_MODE` | `embedded` |
| `MAX_WORKERS` | `2`; use `1` to reduce concurrent CPU/RAM use |
| `ENABLE_DEMO` | `true`; disposable demo workspaces |
| `JWT_SECRET` | Generated and persisted in the data volume when omitted |
| `OPENAI_API_KEY` | Optional; grounded chat/report explanations |
| `OPENAI_MODEL` | `gpt-4o-mini` |

For `docker run`, pass environment variables with `-e` before `analytiq:latest`. Compose reads the optional settings from your root `.env`. Local `.env` files are not baked into the image.

## Logs and health

```bash
docker logs -f analytiq
docker inspect --format='{{.State.Health.Status}}' analytiq
```

Allow initial startup to finish before the health status changes from `starting` to `healthy`.

## Browser verification against the container

With the container running, execute from `frontend` in PowerShell:

```powershell
$env:PLAYWRIGHT_BASE_URL = "http://localhost:8080"
npx playwright install chromium
npm run test:e2e
```

The three desktop, mobile, and clustering scenarios can target development or the packaged production frontend. They cover the guided training wizard, cross-validation comparison, SHAP, predictions, downloads, and responsive visualizations. The desktop workflow also reloads a client-side route and verifies API documentation links. See [the verification record](VERIFICATION.md) for the completed checks.

## Stop, restart, and rebuild

```bash
docker stop analytiq
docker start analytiq
```

After changing source files, rebuild and replace the container while retaining the volume:

```bash
docker build -t analytiq:latest .
docker stop analytiq
docker rm analytiq
docker run -d --name analytiq -p 8080:8080 -v analytiq-data:/app/data --restart unless-stopped analytiq:latest
```

Accounts, datasets, and completed models remain in the named volume. Running model training is stopped when its container stops; incomplete runs are recovered as interrupted jobs by the worker's stale-run detection.

## Move the image to another machine

```bash
docker save -o analytiq-image.tar analytiq:latest
```

On the destination machine:

```bash
docker load -i analytiq-image.tar
docker run -d --name analytiq -p 8080:8080 -v analytiq-data:/app/data --restart unless-stopped analytiq:latest
```

The image contains the application. Your existing accounts and datasets live in the data volume and are not included by `docker save`.
