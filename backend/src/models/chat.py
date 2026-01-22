"""Chat session and message models for RAG chatbot."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.models.user import User


class MessageRole(str, Enum):
    """Chat message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatSession(Base, UUIDMixin, TimestampMixin):
    """Chat session model for storing conversation sessions."""

    __tablename__ = "chat_sessions"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    context_chapter: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User | None"] = relationship(
        "User",
        back_populates="chat_sessions",
        lazy="selectin",
    )
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

    def to_dict(self) -> dict:
        """Convert to API response dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "context_chapter": self.context_chapter,
            "created_at": self.created_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat(),
        }


class ChatMessage(Base, UUIDMixin, TimestampMixin):
    """Chat message model for storing individual messages."""

    __tablename__ = "chat_messages"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    citations: Mapped[list[dict] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
    )

    # Relationships
    session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        back_populates="messages",
    )

    def to_dict(self) -> dict:
        """Convert to API response dictionary."""
        return {
            "id": str(self.id),
            "role": self.role,
            "content": self.content,
            "citations": self.citations or [],
            "created_at": self.created_at.isoformat(),
        }
