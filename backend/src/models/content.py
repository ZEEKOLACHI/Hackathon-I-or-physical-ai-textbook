"""Content variant models for personalization and translation."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, UUIDMixin

if TYPE_CHECKING:
    from src.models.user import User


class VariantType(str, Enum):
    """Content variant type enumeration."""

    PERSONALIZED = "personalized"
    TRANSLATED = "translated"


class ContentVariant(Base, UUIDMixin):
    """Model for personalized or translated content variants."""

    __tablename__ = "content_variants"
    __table_args__ = (
        UniqueConstraint(
            "chapter_id",
            "user_id",
            "variant_type",
            "variant_key",
            name="uq_content_variant_lookup",
        ),
    )

    chapter_id: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    variant_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    variant_key: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User | None"] = relationship(
        "User",
        lazy="selectin",
    )

    def to_dict(self) -> dict:
        """Convert to API response dictionary."""
        return {
            "id": str(self.id),
            "chapter_id": self.chapter_id,
            "variant_type": self.variant_type,
            "variant_key": self.variant_key,
            "content": self.content,
            "is_rtl": self.variant_key == "urdu",
            "created_at": self.created_at.isoformat(),
        }
