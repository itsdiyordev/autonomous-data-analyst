from datetime import timedelta

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash
from sqlalchemy.orm import Session

from .config import settings
from .db import User, get_db, now

password_hasher = PasswordHash.recommended()
bearer = HTTPBearer(auto_error=False)


def make_token(user: User) -> str:
    return jwt.encode(
        {"sub": user.id, "exp": now() + timedelta(hours=24), "iat": now()},
        settings.jwt_secret,
        algorithm="HS256",
    )


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(401, "Please sign in to continue.")
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
        user = db.get(User, payload["sub"])
        if user is None:
            raise ValueError("Unknown user")
        return user
    except (jwt.PyJWTError, ValueError, KeyError):
        raise HTTPException(401, "Your session expired. Please sign in again.") from None
