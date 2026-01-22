"""User profile management routes."""

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.api.dependencies import CurrentUser
from src.db.postgres import get_session
from src.services.auth_service import update_user_profile

router = APIRouter()


# --- Request/Response Schemas ---


class UserProfileResponse(BaseModel):
    """User profile response with full details."""

    id: str
    email: str
    programming_level: str
    robotics_level: str
    hardware_available: list[str]
    created_at: str
    updated_at: str


class UserProfileUpdateRequest(BaseModel):
    """Request to update user profile."""

    programming_level: str | None = Field(
        None,
        pattern="^(none|beginner|intermediate|advanced)$",
    )
    robotics_level: str | None = Field(
        None,
        pattern="^(none|beginner|intermediate|advanced)$",
    )
    hardware_available: list[str] | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    message: str


# --- User Endpoints ---


@router.get(
    "/users/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
    responses={
        401: {"model": ErrorResponse},
    },
)
async def get_current_user_profile(current_user: CurrentUser) -> UserProfileResponse:
    """Get the current user's full profile."""
    return UserProfileResponse(**current_user.to_profile_dict())


@router.patch(
    "/users/me",
    response_model=UserProfileResponse,
    summary="Update current user profile",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
    },
)
async def update_current_user_profile(
    request: UserProfileUpdateRequest,
    current_user: CurrentUser,
) -> UserProfileResponse:
    """Update the current user's background information."""
    async with get_session() as db:
        # Reattach user to this session
        await db.merge(current_user)

        updated_user = await update_user_profile(
            db=db,
            user=current_user,
            programming_level=request.programming_level,
            robotics_level=request.robotics_level,
            hardware_available=request.hardware_available,
        )

        return UserProfileResponse(**updated_user.to_profile_dict())
