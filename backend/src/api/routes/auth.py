"""Authentication routes for signup, signin, signout, and session management."""

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field

from src.api.dependencies import SESSION_COOKIE_NAME, CurrentUser, OptionalUser
from src.db.postgres import get_session
from src.services.auth_service import (
    authenticate_user,
    create_session,
    create_user,
    invalidate_session,
)

router = APIRouter()


# --- Request/Response Schemas ---


class SignUpRequest(BaseModel):
    """Request to sign up a new user."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    programming_level: str = Field("beginner", pattern="^(none|beginner|intermediate|advanced)$")
    robotics_level: str = Field("beginner", pattern="^(none|beginner|intermediate|advanced)$")
    hardware_available: list[str] = Field(default_factory=list)


class SignInRequest(BaseModel):
    """Request to sign in an existing user."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response."""

    id: str
    email: str
    created_at: str


class SessionResponse(BaseModel):
    """Session response."""

    id: str
    expires_at: str
    created_at: str


class AuthResponse(BaseModel):
    """Authentication response with user and session."""

    user: UserResponse
    session: SessionResponse


class SessionInfoResponse(BaseModel):
    """Session info response for /auth/session."""

    user: UserResponse | None
    session: SessionResponse | None


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    message: str


# --- Helper Functions ---


def set_session_cookie(response: Response, token: str) -> None:
    """Set the session cookie on a response."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,  # Requires HTTPS in production
        samesite="lax",
        max_age=30 * 24 * 60 * 60,  # 30 days
    )


def clear_session_cookie(response: Response) -> None:
    """Clear the session cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        secure=True,
        samesite="lax",
    )


# --- Auth Endpoints ---


@router.post(
    "/auth/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def signup(request: SignUpRequest, response: Response) -> AuthResponse:
    """Create a new user account with email, password, and background profile."""
    async with get_session() as db:
        try:
            user = await create_user(
                db=db,
                email=request.email,
                password=request.password,
                programming_level=request.programming_level,
                robotics_level=request.robotics_level,
                hardware_available=request.hardware_available,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": "conflict", "message": str(e)},
            )

        # Create session
        session = await create_session(db, user.id)

        # Set cookie
        set_session_cookie(response, session.token)

        return AuthResponse(
            user=UserResponse(**user.to_dict()),
            session=SessionResponse(**session.to_dict()),
        )


@router.post(
    "/auth/signin",
    response_model=AuthResponse,
    summary="Sign in existing user",
    responses={
        401: {"model": ErrorResponse},
    },
)
async def signin(request: SignInRequest, response: Response) -> AuthResponse:
    """Authenticate user with email and password."""
    async with get_session() as db:
        user = await authenticate_user(db, request.email, request.password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "invalid_credentials", "message": "Invalid email or password"},
            )

        # Create session
        session = await create_session(db, user.id)

        # Set cookie
        set_session_cookie(response, session.token)

        return AuthResponse(
            user=UserResponse(**user.to_dict()),
            session=SessionResponse(**session.to_dict()),
        )


@router.post(
    "/auth/signout",
    status_code=status.HTTP_200_OK,
    summary="Sign out current user",
    responses={
        401: {"model": ErrorResponse},
    },
)
async def signout(
    response: Response,
    current_user: CurrentUser,
) -> dict[str, str]:
    """Invalidate the current session."""
    # Note: We get the token from the cookie via the dependency
    # The actual invalidation happens here
    async with get_session() as db:
        # Get token from user's sessions (most recent)
        if current_user.sessions:
            for session in current_user.sessions:
                await invalidate_session(db, session.token)

    # Clear cookie
    clear_session_cookie(response)

    return {"message": "Successfully signed out"}


@router.get(
    "/auth/session",
    response_model=SessionInfoResponse,
    summary="Get current session",
)
async def get_session_info(current_user: OptionalUser) -> SessionInfoResponse:
    """Return the current user if authenticated."""
    if not current_user:
        return SessionInfoResponse(user=None, session=None)

    # Get most recent active session
    active_session = None
    if current_user.sessions:
        for session in current_user.sessions:
            if not session.is_expired:
                active_session = session
                break

    return SessionInfoResponse(
        user=UserResponse(**current_user.to_dict()),
        session=SessionResponse(**active_session.to_dict()) if active_session else None,
    )
