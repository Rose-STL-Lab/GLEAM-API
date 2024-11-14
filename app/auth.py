from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.db import check_api_key, get_user_from_api_key

api_key_header = APIKeyHeader(name='X-API-Key')


def get_user(api_key_header: str = Security(api_key_header)):
    user = get_user_from_api_key(api_key_header)
    if len(user) != 0:
        return user[0]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key"
    )