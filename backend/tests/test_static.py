import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.static import SPAStaticFiles


@pytest.fixture
def web_client(tmp_path):
    public = tmp_path / "public"
    (public / "assets").mkdir(parents=True)
    (public / "index.html").write_text("<html><body>Analytiq dashboard</body></html>")
    (public / "assets" / "app-a1b2.js").write_text("console.log('ready')")
    (tmp_path / "secret.txt").write_text("private file")
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    app.mount("/", SPAStaticFiles(directory=public, html=True))
    with TestClient(app) as client:
        yield client


def test_dashboard_deep_links_and_api_routes(web_client):
    for path in ("/", "/datasets", "/datasets/abc123", "/runs/abc123", "/settings"):
        response = web_client.get(path)
        assert response.status_code == 200
        assert "Analytiq dashboard" in response.text
        assert response.headers["cache-control"] == "no-cache"
    assert web_client.get("/api/health").json() == {"status": "ok"}
    assert web_client.get("/api/not-an-endpoint").status_code == 404
    assert web_client.get("/api/not-an-endpoint").headers["content-type"].startswith("application/json")
    assert web_client.post("/datasets").status_code == 405


def test_assets_are_cached_and_missing_files_stay_404(web_client):
    response = web_client.get("/assets/app-a1b2.js")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]
    unchanged = web_client.get("/assets/app-a1b2.js", headers={"If-None-Match": response.headers["etag"]})
    assert unchanged.status_code == 304
    assert "immutable" in unchanged.headers["cache-control"]
    assert web_client.get("/assets/missing.js").status_code == 404
    assert web_client.get("/missing.png").status_code == 404


def test_static_paths_cannot_read_outside_frontend(web_client):
    response = web_client.get("/%2e%2e/secret.txt")
    assert response.status_code == 404
    assert "private file" not in response.text
