"""User and session models for authentication."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.models.chat import ChatSession


class ExperienceLevel(str, Enum):
    """Experience level enumeration."""

    NONE = "none"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class User(Base, UUIDMixin, TimestampMixin):
    """User model for authentication and profile."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Background profile fields
    programming_level: Mapped[str] = mapped_column(
        String(20),
        default=ExperienceLevel.BEGINNER.value,
        nullable=False,
    )
    robotics_level: Mapped[str] = mapped_column(
        String(20),
        default=ExperienceLevel.BEGINNER.value,
        nullable=False,
    )
    hardware_available: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        default=list,
    )

    # Relationships
    sessions: Mapped[list["Session"]] = relationship(
        "Session",
        back_populates="user",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="user",
        lazy="selectin",
    )

    def to_dict(self) -> dict:
        """Convert to API response dictionary (basic info)."""
        return {
            "id": str(self.id),
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }

    def to_profile_dict(self) -> dict:
        """Convert to API response dictionary (full profile)."""
        return {
            "id": str(self.id),
            "email": self.email,
            "programming_level": self.programming_level,
            "robotics_level": self.robotics_level,
            "hardware_available": self.hardware_available or [],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Session(Base, UUIDMixin):
    """Session model for authentication sessions."""

    __tablename__ = "sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="sessions",
    )

    def to_dict(self) -> dict:
        """Convert to API response dictionary."""
        return {
            "id": str(self.id),
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(self.expires_at.tzinfo) > self.expires_at
