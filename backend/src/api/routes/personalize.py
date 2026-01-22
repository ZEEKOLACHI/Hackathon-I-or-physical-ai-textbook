"""Content personalization routes."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.api.dependencies import CurrentUser
from src.db.postgres import get_session
from src.services.personalization_service import personalize_chapter

router = APIRouter()


# --- Request/Response Schemas ---


class PersonalizeRequest(BaseModel):
    """Request to personalize chapter content."""

    chapter_id: str


class ContentVariantResponse(BaseModel):
    """Personalized content variant response."""

    id: str
    chapter_id: str
    variant_type: str
    variant_key: str
    content: str
    is_rtl: bool
    created_at: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    message: str


# --- Personalization Endpoints ---


@router.post(
    "/content/personalize",
    response_model=ContentVariantResponse,
    summary="Personalize chapter content",
    description=(
        "Generates personalized chapter content based on user's background. "
        "Returns cached version if available."
    ),
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def personalize_content(
    request: PersonalizeRequest,
    current_user: CurrentUser,
) -> ContentVariantResponse:
    """
    Personalize chapter content for the current user.

    The content is adapted based on the user's programming and robotics
    experience levels, as well as available hardware.
    """
    async with get_session() as db:
        # Reattach user to this session
        merged_user = await db.merge(current_user)

        variant = await personalize_chapter(
            db=db,
            chapter_id=request.chapter_id,
            user=merged_user,
        )

        if variant is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "not_found", "message": f"Chapter '{request.chapter_id}' not found"},
            )

        return ContentVariantResponse(**variant.to_dict())
