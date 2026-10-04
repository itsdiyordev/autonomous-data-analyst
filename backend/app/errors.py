"""Public error contracts; unexpected exception details stay in server logs."""
import json

from fastapi.responses import JSONResponse


class AnalysisError(ValueError):
    """An expected analytical failure with a safe, actionable message."""

    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def error_response(status, code, message, details=None, headers=None, legacy_detail=None):
    error = {"code": code, "message": message, "details": details or {}}
    # Keep the existing FastAPI detail contract for clients migrating incrementally.
    return JSONResponse({"error": error, "detail": message if legacy_detail is None else legacy_detail}, status_code=status, headers=headers)


def encode_failure(exc):
    error = ({"code": exc.code, "message": exc.message, "details": exc.details} if isinstance(exc, AnalysisError) else
             {"code": "ANALYSIS_FAILED", "message": "The analysis could not be completed. Please retry or review the dataset and configuration.", "details": {}})
    return json.dumps(error)


def public_failure(value):
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict) and isinstance(parsed.get("code"), str) and isinstance(parsed.get("message"), str) and isinstance(parsed.get("details"), dict):
            return parsed
    except (TypeError, ValueError):
        pass
    # Older database rows may contain raw exception strings; never redisplay them.
    return {"code": "ANALYSIS_FAILED", "message": "This analysis failed. Please rerun it to obtain a current diagnostic.", "details": {}}
