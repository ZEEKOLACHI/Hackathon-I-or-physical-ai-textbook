"""Personalization service for adapting chapter content to user background."""

import os
import uuid
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.content import ContentVariant, VariantType
from src.models.user import User

# Paths for chapter content
BOOK_DOCS_PATH = Path(__file__).parent.parent.parent.parent / "book" / "docs"

# Chapter ID to file path mapping (ch-{part}-{chapter} format)
PART_DIRS = {
    1: "part-1-foundations",
    2: "part-2-perception",
    3: "part-3-planning",
    4: "part-4-control",
    5: "part-5-learning",
    6: "part-6-humanoids",
    7: "part-7-integration",
}

CHAPTER_FILES = {
    "ch-1-01": "01-introduction.md",
    "ch-1-02": "02-ros2-fundamentals.md",
    "ch-1-03": "03-simulation-basics.md",
    "ch-2-04": "04-computer-vision.md",
    "ch-2-05": "05-sensor-fusion.md",
    "ch-2-06": "06-3d-perception.md",
    "ch-3-07": "07-motion-planning.md",
    "ch-3-08": "08-task-planning.md",
    "ch-3-09": "09-behavior-trees.md",
    "ch-4-10": "10-pid-control.md",
    "ch-4-11": "11-force-control.md",
    "ch-4-12": "12-whole-body-control.md",
    "ch-5-13": "13-reinforcement-learning.md",
    "ch-5-14": "14-imitation-learning.md",
    "ch-5-15": "15-vla-models.md",
    "ch-6-16": "16-humanoid-kinematics.md",
    "ch-6-17": "17-bipedal-locomotion.md",
    "ch-6-18": "18-manipulation.md",
    "ch-7-19": "19-system-integration.md",
    "ch-7-20": "20-safety-standards.md",
    "ch-7-21": "21-future-directions.md",
}

# LLM configuration
PERSONALIZATION_MODEL = "gpt-4o-mini"
MAX_TOKENS = 4096
TEMPERATURE = 0.7

# Personalization prompt template
PERSONALIZATION_PROMPT = """You are an expert educational content adapter. Your task is to personalize the following textbook chapter content for a specific learner's background.

## Learner Profile
- **Programming Experience**: {programming_level}
- **Robotics Experience**: {robotics_level}
- **Available Hardware**: {hardware}

## Personalization Guidelines

Based on the learner's background, adapt the content by:

1. **For Beginners (none/beginner level)**:
   - Add more explanations for technical terms
   - Include additional context and analogies
   - Break down complex concepts into smaller steps
   - Add "Why this matters" sections
   - Include more code comments

2. **For Intermediate learners**:
   - Maintain the current level of detail
   - Add connections to related concepts
   - Include practical tips and common pitfalls

3. **For Advanced learners**:
   - Add deeper technical insights
   - Include references to research papers or advanced topics
   - Suggest optimization techniques
   - Add challenging exercises or extensions

4. **Hardware-specific adaptations**:
   - If learner has specific hardware (e.g., Jetson Nano, TurtleBot), add relevant examples
   - If simulation_only, emphasize simulation-based approaches

## Rules
- Preserve all code blocks exactly as they are (syntax must remain valid)
- Maintain the markdown structure (headings, lists, tables)
- Keep the core educational content accurate
- Add personalized sections with clear markers like "💡 For your level:" or "🔧 With your hardware:"
- Do not remove any essential information

## Original Chapter Content

{content}

## Output

Return the personalized chapter content in markdown format. Ensure it flows naturally and integrates personalization seamlessly."""


# OpenAI client
_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Get or create OpenAI client instance."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def get_chapter_path(chapter_id: str) -> Path | None:
    """Get the file path for a chapter ID."""
    if chapter_id not in CHAPTER_FILES:
        return None

    # Extract part number from chapter_id (e.g., ch-1-01 -> 1)
    parts = chapter_id.split("-")
    if len(parts) != 3:
        return None

    try:
        part_num = int(parts[1])
    except ValueError:
        return None

    if part_num not in PART_DIRS:
        return None

    part_dir = PART_DIRS[part_num]
    filename = CHAPTER_FILES[chapter_id]
    return BOOK_DOCS_PATH / part_dir / filename


async def read_chapter_content(chapter_id: str) -> str | None:
    """Read the markdown content for a chapter."""
    path = get_chapter_path(chapter_id)
    if path is None or not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def compute_variant_key(user: User) -> str:
    """Compute a variant key based on user's background profile."""
    # Combine programming and robotics levels for the key
    return f"{user.programming_level}_{user.robotics_level}"


async def get_cached_variant(
    db: AsyncSession,
    chapter_id: str,
    user_id: uuid.UUID,
    variant_key: str,
) -> ContentVariant | None:
    """Check if a personalized variant is already cached."""
    result = await db.execute(
        select(ContentVariant).where(
            ContentVariant.chapter_id == chapter_id,
            ContentVariant.user_id == user_id,
            ContentVariant.variant_type == VariantType.PERSONALIZED.value,
            ContentVariant.variant_key == variant_key,
        )
    )
    return result.scalar_one_or_none()


async def generate_personalized_content(
    original_content: str,
    user: User,
) -> str:
    """Generate personalized content using LLM."""
    client = get_openai_client()

    # Format hardware list
    hardware = ", ".join(user.hardware_available) if user.hardware_available else "simulation_only"

    prompt = PERSONALIZATION_PROMPT.format(
        programming_level=user.programming_level,
        robotics_level=user.robotics_level,
        hardware=hardware,
        content=original_content,
    )

    response = await client.chat.completions.create(
        model=PERSONALIZATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert educational content adapter specializing in robotics and AI education.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content or original_content


async def personalize_chapter(
    db: AsyncSession,
    chapter_id: str,
    user: User,
) -> ContentVariant | None:
    """
    Personalize chapter content for a user.

    Returns cached version if available, otherwise generates and caches new version.

    Args:
        db: Database session
        chapter_id: Chapter identifier (e.g., "ch-1-01")
        user: User requesting personalization

    Returns:
        ContentVariant with personalized content, or None if chapter not found
    """
    # Read original chapter content
    original_content = await read_chapter_content(chapter_id)
    if original_content is None:
        return None

    # Check for cached variant
    variant_key = compute_variant_key(user)
    cached = await get_cached_variant(db, chapter_id, user.id, variant_key)
    if cached is not None:
        return cached

    # Generate personalized content
    personalized_content = await generate_personalized_content(original_content, user)

    # Create and save new variant
    variant = ContentVariant(
        chapter_id=chapter_id,
        user_id=user.id,
        variant_type=VariantType.PERSONALIZED.value,
        variant_key=variant_key,
        content=personalized_content,
    )
    db.add(variant)
    await db.commit()
    await db.refresh(variant)

    return variant
