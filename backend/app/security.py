"""Streaming request limits apply before JSON/multipart parsers allocate a body."""
from starlette.exceptions import HTTPException
from starlette.datastructures import Headers

from .config import settings
from .errors import error_response


class BodySizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = ((settings.max_upload_mb + 2) * 1024 * 1024 if scope["path"].rstrip("/") == "/api/datasets" and scope["method"] == "POST" else 2 * 1024 * 1024)
        length = Headers(scope=scope).get("content-length")
        if length is not None and (not length.isascii() or not length.isdecimal() or int(length) > limit):
            response = error_response(413, "REQUEST_TOO_LARGE", "Request body exceeds the allowed size.")
            return await response(scope, receive, send)
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise HTTPException(413, "Request body exceeds the allowed size.")
            return message

        await self.app(scope, bounded_receive, send)
