"""Translation service for translating chapter content to Urdu."""

import re
from pathlib import Path

import google.generativeai as genai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.content import ContentVariant, VariantType

# Paths for chapter content (shared with personalization service)
BOOK_DOCS_PATH = Path(__file__).parent.parent.parent.parent / "book" / "docs"

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
TRANSLATION_MODEL = "gemini-1.5-flash"
MAX_TOKENS = 4096
TEMPERATURE = 0.3  # Lower temperature for more accurate translation

# Code block placeholder pattern
CODE_BLOCK_PLACEHOLDER = "<<<CODE_BLOCK_{index}>>>"
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# Translation prompt template
TRANSLATION_PROMPT = """You are an expert translator specializing in technical content translation from English to Urdu.

## Task
Translate the following technical textbook content from English to Urdu.

## Rules
1. **Preserve all placeholders**: Any text like <<<CODE_BLOCK_0>>>, <<<CODE_BLOCK_1>>>, etc. must remain EXACTLY as is - do not translate or modify these.
2. **Technical terms**: Keep technical terms (like ROS, Python, API, SDK, etc.) in English but you may add Urdu transliteration in parentheses if helpful.
3. **Markdown formatting**: Preserve all markdown syntax (headings #, lists -, bold **, links [], tables, etc.)
4. **Accuracy**: Ensure the translation is accurate and natural-sounding in Urdu.
5. **Educational tone**: Maintain the educational and informative tone of the original.
6. **Numbers and symbols**: Keep numbers, mathematical symbols, and special characters as-is.

## Content to Translate

{content}

## Output

Return ONLY the translated content in Urdu. Do not add any explanations or notes."""


# Google AI state
_initialized = False


def _init_google_ai() -> None:
    """Initialize Google AI client."""
    global _initialized
    if not _initialized:
        genai.configure(api_key=settings.google_api_key)
        _initialized = True


def get_chapter_path(chapter_id: str) -> Path | None:
    """Get the file path for a chapter ID."""
    if chapter_id not in CHAPTER_FILES:
        return None

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


def extract_code_blocks(content: str) -> tuple[str, list[str]]:
    """
    Extract code blocks from content and replace with placeholders.

    Returns:
        Tuple of (content with placeholders, list of extracted code blocks)
    """
    code_blocks: list[str] = []

    def replace_with_placeholder(match: re.Match) -> str:
        index = len(code_blocks)
        code_blocks.append(match.group(0))
        return CODE_BLOCK_PLACEHOLDER.format(index=index)

    content_with_placeholders = CODE_BLOCK_PATTERN.sub(replace_with_placeholder, content)
    return content_with_placeholders, code_blocks


def restore_code_blocks(content: str, code_blocks: list[str]) -> str:
    """Restore code blocks from placeholders."""
    result = content
    for index, code_block in enumerate(code_blocks):
        placeholder = CODE_BLOCK_PLACEHOLDER.format(index=index)
        result = result.replace(placeholder, code_block)
    return result


async def get_cached_translation(
    db: AsyncSession,
    chapter_id: str,
    target_language: str,
) -> ContentVariant | None:
    """Check if a translation is already cached."""
    result = await db.execute(
        select(ContentVariant).where(
            ContentVariant.chapter_id == chapter_id,
            ContentVariant.user_id.is_(None),  # Translations are not user-specific
            ContentVariant.variant_type == VariantType.TRANSLATED.value,
            ContentVariant.variant_key == target_language,
        )
    )
    return result.scalar_one_or_none()


async def translate_content(
    content: str,
    target_language: str,
) -> str:
    """Translate content using LLM while preserving code blocks."""
    _init_google_ai()

    # Extract code blocks
    content_with_placeholders, code_blocks = extract_code_blocks(content)

    # Translate the content (with placeholders)
    prompt = TRANSLATION_PROMPT.format(content=content_with_placeholders)

    model = genai.GenerativeModel(
        TRANSLATION_MODEL,
        generation_config=genai.GenerationConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        ),
    )

    response = model.generate_content(prompt)

    translated_with_placeholders = response.text or content_with_placeholders

    # Restore code blocks
    translated_content = restore_code_blocks(translated_with_placeholders, code_blocks)

    return translated_content


async def translate_chapter(
    db: AsyncSession,
    chapter_id: str,
    target_language: str = "urdu",
) -> ContentVariant | None:
    """
    Translate chapter content to the target language.

    Returns cached version if available, otherwise translates and caches.

    Args:
        db: Database session
        chapter_id: Chapter identifier (e.g., "ch-1-01")
        target_language: Target language (currently only "urdu" supported)

    Returns:
        ContentVariant with translated content, or None if chapter not found
    """
    # Validate target language
    if target_language.lower() != "urdu":
        return None

    target_language = target_language.lower()

    # Read original chapter content
    original_content = await read_chapter_content(chapter_id)
    if original_content is None:
        return None

    # Check for cached translation
    cached = await get_cached_translation(db, chapter_id, target_language)
    if cached is not None:
        return cached

    # Translate content
    translated_content = await translate_content(original_content, target_language)

    # Create and save new variant
    variant = ContentVariant(
        chapter_id=chapter_id,
        user_id=None,  # Translations are shared across users
        variant_type=VariantType.TRANSLATED.value,
        variant_key=target_language,
        content=translated_content,
    )
    db.add(variant)
    await db.commit()
    await db.refresh(variant)

    return variant
