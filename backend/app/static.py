from pathlib import PurePosixPath

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


class SPAStaticFiles(StaticFiles):
    """Serve the built dashboard, preserving API errors and browser deep links."""

    async def get_response(self, path, scope):
        parts = PurePosixPath(path).parts
        if parts and parts[0] in {"api", "docs", "redoc", "openapi.json"}:
            raise HTTPException(404)
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            # Missing assets stay 404; client-side page routes receive the SPA shell.
            if exc.status_code != 404 or scope["method"] not in {"GET", "HEAD"}:
                raise
            if parts and (parts[0] == "assets" or "." in parts[-1] or ".." in parts):
                raise
            response = await super().get_response("index.html", scope)
        if response.status_code in {200, 304}:
            response.headers["Cache-Control"] = (
                "public, max-age=31536000, immutable" if parts and parts[0] == "assets" else "no-cache"
            )
        return response
