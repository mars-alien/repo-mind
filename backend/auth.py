"""
JWT + bcrypt authentication utilities.
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

SECRET_KEY  = os.environ.get("JWT_SECRET", "change-this-in-production-32-chars-min")
ALGORITHM   = "HS256"
EXPIRE_DAYS = 30          # token valid for 30 days

bearer = HTTPBearer()


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
) -> str:
    """FastAPI dependency — returns the user_id from a valid JWT Bearer token."""
    payload = decode_token(creds.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token has no subject.")
    return user_id
