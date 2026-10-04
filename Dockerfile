# Build the production dashboard; Node.js is not needed in the final image.
FROM node:24-alpine AS web-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV NODE_OPTIONS=--max-old-space-size=1024
RUN npm run build

FROM python:3.12-slim AS python-base

FROM python-base AS python-build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --no-install-project --compile-bytecode

# One runtime container serves the dashboard, API, and background ML processes.
FROM python-base AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 analyst \
    && mkdir -p /app/data \
    && chown analyst:analyst /app/data
WORKDIR /app/backend
COPY --from=python-build /app/backend/.venv /app/backend/.venv
COPY backend/app ./app
COPY --from=web-build /web/dist /app/frontend/dist
ENV PATH="/app/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    DATABASE_URL=sqlite:////app/data/analytiq.db \
    STATIC_DIR=/app/frontend/dist \
    WORKER_MODE=embedded \
    MAX_WORKERS=2 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
USER analyst
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4)"]
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-graceful-shutdown", "25"]
