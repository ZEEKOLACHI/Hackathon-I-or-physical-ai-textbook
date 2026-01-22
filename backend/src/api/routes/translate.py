"""Content translation routes."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.db.postgres import get_session
from src.services.translation_service import translate_chapter

router = APIRouter()


# --- Request/Response Schemas ---


class TranslateRequest(BaseModel):
    """Request to translate chapter content."""

    chapter_id: str
    target_language: str = Field(
        default="urdu",
        pattern="^urdu$",
        description="Target language (only 'urdu' supported)",
    )


class ContentVariantResponse(BaseModel):
    """Translated content variant response."""

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


# --- Translation Endpoints ---


@router.post(
    "/content/translate",
    response_model=ContentVariantResponse,
    summary="Translate chapter to Urdu",
    description=(
        "Translates chapter content to Urdu while preserving code blocks. "
        "Returns cached version if available. Does not require authentication."
    ),
    responses={
        404: {"model": ErrorResponse},
    },
)
async def translate_content(
    request: TranslateRequest,
) -> ContentVariantResponse:
    """
    Translate chapter content to Urdu.

    Code blocks are preserved in English while all other content is translated.
    Translations are cached for future requests.
    """
    async with get_session() as db:
        variant = await translate_chapter(
            db=db,
            chapter_id=request.chapter_id,
            target_language=request.target_language,
        )

        if variant is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "not_found",
                    "message": f"Chapter '{request.chapter_id}' not found or language not supported",
                },
            )

        return ContentVariantResponse(**variant.to_dict())
