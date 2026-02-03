"""Chat and search endpoints for the RAG chatbot."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.db.postgres import get_session
from src.models.chat import ChatMessage, ChatSession, MessageRole
from src.services.llm_service import generate_response, generate_response_with_selected_text
from src.services.rag_service import (
    format_search_results,
    get_context_for_query,
    search_content,
)

router = APIRouter()


# --- Request/Response Schemas ---


class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""

    context_chapter: str | None = Field(
        None, description="Chapter ID being viewed when starting chat"
    )


class SendMessageRequest(BaseModel):
    """Request to send a message."""

    content: str = Field(..., min_length=1, max_length=2000)
    selected_text: str | None = Field(None, max_length=5000)


class CitationResponse(BaseModel):
    """Citation reference in response."""

    chapter_id: str
    section_id: str
    section_title: str
    relevance_score: float


class MessageResponse(BaseModel):
    """Chat message response."""

    id: str
    role: str
    content: str
    citations: list[CitationResponse]
    created_at: str


class SessionResponse(BaseModel):
    """Chat session response."""

    id: str
    user_id: str | None
    context_chapter: str | None
    created_at: str
    last_message_at: str


class ChatResponse(BaseModel):
    """Response containing message and session."""

    message: MessageResponse
    session: SessionResponse


class MessagesListResponse(BaseModel):
    """Response for listing messages."""

    messages: list[MessageResponse]
    has_more: bool


class SearchResultResponse(BaseModel):
    """Search result item."""

    chunk_id: str
    chapter_id: str
    section_id: str
    section_title: str
    content_preview: str
    has_code: bool
    difficulty: str
    score: float


class SearchResponse(BaseModel):
    """Search response."""

    results: list[SearchResultResponse]
    query: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    message: str
    details: dict | None = None


# --- Helper Functions ---


def session_to_response(session: ChatSession) -> SessionResponse:
    """Convert ChatSession model to response."""
    return SessionResponse(
        id=str(session.id),
        user_id=str(session.user_id) if session.user_id else None,
        context_chapter=session.context_chapter,
        created_at=session.created_at.isoformat(),
        last_message_at=session.last_message_at.isoformat(),
    )


def message_to_response(message: ChatMessage) -> MessageResponse:
    """Convert ChatMessage model to response."""
    citations = [
        CitationResponse(**c) for c in (message.citations or [])
    ]
    return MessageResponse(
        id=str(message.id),
        role=message.role,
        content=message.content,
        citations=citations,
        created_at=message.created_at.isoformat(),
    )


# --- Chat Session Endpoints ---


@router.post(
    "/chat/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
    responses={500: {"model": ErrorResponse}},
)
async def create_chat_session(
    request: CreateSessionRequest | None = None,
) -> SessionResponse:
    """Create a new conversation session."""
    async with get_session() as session:
        chat_session = ChatSession(
            context_chapter=request.context_chapter if request else None,
        )
        session.add(chat_session)
        await session.commit()
        await session.refresh(chat_session)

        return session_to_response(chat_session)


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=MessagesListResponse,
    summary="Get messages in a session",
    responses={404: {"model": ErrorResponse}},
)
async def get_chat_messages(
    session_id: uuid.UUID,
    limit: int = Query(50, le=100),
    before: uuid.UUID | None = None,
) -> MessagesListResponse:
    """Get messages in a chat session."""
    async with get_session() as session:
        # Check session exists
        chat_session = await session.get(ChatSession, session_id)
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "not_found", "message": "Session not found"},
            )

        # Build query
        query = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit + 1)  # Get one extra to check has_more
        )

        if before:
            before_msg = await session.get(ChatMessage, before)
            if before_msg:
                query = query.where(ChatMessage.created_at < before_msg.created_at)

        result = await session.execute(query)
        messages = list(result.scalars().all())

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        # Reverse to get chronological order
        messages.reverse()

        return MessagesListResponse(
            messages=[message_to_response(m) for m in messages],
            has_more=has_more,
        )


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatResponse,
    summary="Send a message and get response",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def send_chat_message(
    session_id: uuid.UUID,
    request: SendMessageRequest,
) -> ChatResponse:
    """Send a user message and get an AI response."""
    async with get_session() as session:
        # Get session with messages
        result = await session.execute(
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(selectinload(ChatSession.messages))
        )
        chat_session = result.scalar_one_or_none()

        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "not_found", "message": "Session not found"},
            )

        # Save user message
        user_message = ChatMessage(
            session_id=session_id,
            role=MessageRole.USER.value,
            content=request.content,
        )
        session.add(user_message)

        # Get context from RAG
        context, citations = await get_context_for_query(
            query=request.content,
            selected_text=request.selected_text,
        )

        # Build conversation history
        history = [
            {"role": m.role, "content": m.content}
            for m in chat_session.messages[-6:]  # Last 6 messages
        ]

        # Generate response
        if request.selected_text:
            response_text = await generate_response_with_selected_text(
                user_message=request.content,
                selected_text=request.selected_text,
                context=context,
            )
        else:
            response_text = await generate_response(
                user_message=request.content,
                context=context,
                conversation_history=history,
            )

        # Save assistant message
        assistant_message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT.value,
            content=response_text,
            citations=citations,
        )
        session.add(assistant_message)

        # Update session timestamp
        chat_session.last_message_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(chat_session)
        await session.refresh(assistant_message)

        return ChatResponse(
            message=message_to_response(assistant_message),
            session=session_to_response(chat_session),
        )


# --- Search Endpoint ---


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search textbook content",
    responses={400: {"model": ErrorResponse}},
)
async def search_textbook_content(
    q: str = Query(..., min_length=3, max_length=500),
    chapter_id: str | None = None,
    difficulty: str | None = Query(None, pattern="^(beginner|intermediate|advanced)$"),
    limit: int = Query(5, le=20),
) -> SearchResponse:
    """Perform semantic search across textbook content."""
    results = await search_content(
        query=q,
        limit=limit,
        chapter_id=chapter_id,
        difficulty=difficulty,
    )

    formatted = format_search_results(results)

    return SearchResponse(
        results=[SearchResultResponse(**r) for r in formatted],
        query=q,
    )


# --- Stateless Chat Endpoint (for serverless) ---


class AskRequest(BaseModel):
    """Request for stateless chat."""

    question: str = Field(..., min_length=1, max_length=2000)
    selected_text: str | None = Field(None, max_length=5000)


class AskResponse(BaseModel):
    """Response for stateless chat."""

    answer: str
    citations: list[CitationResponse]


@router.post(
    "/chat/ask",
    response_model=AskResponse,
    summary="Ask a question (stateless)",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Stateless Q&A endpoint - no session required.
    Ideal for serverless deployments where sessions don't persist.
    """
    # Get context from RAG
    context, citations = await get_context_for_query(
        query=request.question,
        selected_text=request.selected_text,
    )

    # Generate response
    if request.selected_text:
        answer = await generate_response_with_selected_text(
            user_message=request.question,
            selected_text=request.selected_text,
            context=context,
        )
    else:
        answer = await generate_response(
            user_message=request.question,
            context=context,
            conversation_history=[],
        )

    return AskResponse(
        answer=answer,
        citations=[CitationResponse(**c) for c in citations],
    )
