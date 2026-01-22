"""Create content_variants table.

Revision ID: 003
Revises: 002
Create Date: 2026-01-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create content_variants table."""
    op.create_table(
        "content_variants",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chapter_id", sa.String(20), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("variant_type", sa.String(20), nullable=False),
        sa.Column("variant_key", sa.String(50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "chapter_id",
            "user_id",
            "variant_type",
            "variant_key",
            name="uq_content_variant_lookup",
        ),
    )
    op.create_index("ix_content_variants_chapter_id", "content_variants", ["chapter_id"])
    op.create_index("ix_content_variants_user_id", "content_variants", ["user_id"])
    op.create_index(
        "ix_content_variants_lookup",
        "content_variants",
        ["chapter_id", "variant_type", "variant_key"],
    )


def downgrade() -> None:
    """Drop content_variants table."""
    op.drop_index("ix_content_variants_lookup", table_name="content_variants")
    op.drop_index("ix_content_variants_user_id", table_name="content_variants")
    op.drop_index("ix_content_variants_chapter_id", table_name="content_variants")
    op.drop_table("content_variants")
