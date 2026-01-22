"""FastAPI dependencies for authentication and common functionality."""

from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status

from src.db.postgres import get_session
from src.models.user import User
from src.services.auth_service import get_user_by_session_token

# Cookie name for session token
SESSION_COOKIE_NAME = "session_token"


async def get_current_user_optional(
    session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
) -> User | None:
    """
    Get the current user if authenticated, None otherwise.

    This dependency does not raise an error if the user is not authenticated.
    Use this for endpoints that work with or without authentication.
    """
    if not session_token:
        return None

    async with get_session() as db:
        user = await get_user_by_session_token(db, session_token)
        return user


async def get_current_user(
    user: Annotated[User | None, Depends(get_current_user_optional)],
) -> User:
    """
    Get the current authenticated user.

    Raises HTTPException 401 if not authenticated.
    Use this for endpoints that require authentication.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Authentication required"},
        )
    return user


# Type aliases for dependency injection
OptionalUser = Annotated[User | None, Depends(get_current_user_optional)]
CurrentUser = Annotated[User, Depends(get_current_user)]
