from pathlib import Path

from fastapi import HTTPException

from .config import settings


def artifact_file(directory, filename):
    path = (Path(directory) / filename).resolve()
    root = (settings.data_dir / "models").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "The requested server-generated artifact was not found.")
    return path
