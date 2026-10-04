FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
RUN useradd --create-home --uid 10001 analyst && mkdir -p /app/data && chown -R analyst:analyst /app
ENV PATH="/app/.venv/bin:$PATH" DATA_DIR=/app/data PYTHONUNBUFFERED=1 WORKER_MODE=external
USER analyst
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
