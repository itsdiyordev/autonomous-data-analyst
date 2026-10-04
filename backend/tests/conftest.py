import os
import tempfile

import pytest
from fastapi.testclient import TestClient

test_directory = tempfile.TemporaryDirectory(prefix="analytiq-tests-", dir="/tmp/omnirush" if os.path.isdir("/tmp/omnirush") else None)
os.environ["DATA_DIR"] = test_directory.name
os.environ["DATABASE_URL"] = f"sqlite:///{test_directory.name}/test.db"
os.environ["WORKER_MODE"] = "external"
os.environ["OPENAI_API_KEY"] = ""

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as value:
        yield value


@pytest.fixture
def auth(client):
    import secrets
    from app.main import rate_buckets
    # Each fixture is an independent simulated user session; isolate rate state.
    # Dedicated security tests exercise exhaustion inside one fixture/session.
    rate_buckets.clear()
    response = client.post("/api/auth/register", json={"email": f"user-{secrets.token_hex(4)}@example.com", "name": "Test Analyst", "password": "correct-horse-2026"})
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}
