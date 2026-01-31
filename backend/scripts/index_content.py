#!/usr/bin/env python3
"""Content indexing script for the Physical AI textbook.

This script reads all markdown chapters, splits them into chunks,
generates embeddings, and stores them in Qdrant for RAG retrieval.

Usage:
    python -m scripts.index_content [--force]

Options:
    --force     Re-index all content even if already indexed
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.vector_store import (  # noqa: E402
    get_collection_info,
    init_store,
    upsert_vectors,
)
from src.services.embedding_service import chunk_text, get_embeddings  # noqa: E402

# Configuration
BOOK_DOCS_PATH = Path(__file__).parent.parent.parent / "book" / "docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    if not content.startswith("---"):
        return {}, content

    # Find end of frontmatter
    end_match = re.search(r"\n---\n", content[3:])
    if not end_match:
        return {}, content

    frontmatter_str = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body


def extract_sections(content: str) -> list[dict[str, Any]]:
    """
    Extract sections from markdown content.

    Args:
        content: Markdown body content

    Returns:
        List of section dictionaries with id, title, and content
    """
    sections = []
    current_section = {"id": "intro", "title": "Introduction", "content": ""}
    lines = content.split("\n")

    for line in lines:
        # Check for headers (## or ###)
        header_match = re.match(r"^(#{2,3})\s+(.+)$", line)
        if header_match:
            # Save previous section if it has content
            if current_section["content"].strip():
                sections.append(current_section)

            # Start new section
            title = header_match.group(2).strip()
            section_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            current_section = {
                "id": section_id,
                "title": title,
                "content": "",
            }
        else:
            current_section["content"] += line + "\n"

    # Don't forget the last section
    if current_section["content"].strip():
        sections.append(current_section)

    return sections


def detect_code_blocks(content: str) -> bool:
    """Check if content contains code blocks."""
    return "```" in content


async def index_chapter(
    chapter_path: Path,
    force: bool = False,
) -> int:
    """
    Index a single chapter file.

    Args:
        chapter_path: Path to the markdown file
        force: Whether to re-index existing content

    Returns:
        Number of chunks indexed
    """
    print(f"Processing: {chapter_path.name}")

    # Read file content
    content = chapter_path.read_text(encoding="utf-8")

    # Parse frontmatter
    frontmatter, body = parse_frontmatter(content)

    chapter_id = frontmatter.get("id", chapter_path.stem)
    chapter_title = frontmatter.get("title", chapter_path.stem)
    difficulty = frontmatter.get("difficulty", "intermediate")

    # Extract sections
    sections = extract_sections(body)

    all_chunks = []
    all_payloads = []

    for section in sections:
        section_content = section["content"].strip()
        if not section_content:
            continue

        # Split section into chunks
        chunks = chunk_text(section_content, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_payloads.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_title": chapter_title,
                    "section_id": section["id"],
                    "section_title": section["title"],
                    "content": chunk,
                    "has_code": detect_code_blocks(chunk),
                    "difficulty": difficulty,
                    "chunk_index": i,
                    "source_file": chapter_path.name,
                }
            )

    if not all_chunks:
        print(f"  No content to index in {chapter_path.name}")
        return 0

    # Generate embeddings
    print(f"  Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = await get_embeddings(all_chunks)

    print(f"  Upserting to vector store...")
    await upsert_vectors(embeddings, all_payloads)

    print(f"  Indexed {len(all_chunks)} chunks from {chapter_path.name}")
    return len(all_chunks)


async def index_all_chapters(force: bool = False) -> None:
    """
    Index all chapters in the book docs directory.

    Args:
        force: Whether to re-index existing content
    """
    print("=" * 60)
    print("Physical AI Textbook Content Indexer")
    print("=" * 60)

    # Initialize vector store
    print("\nInitializing vector store...")
    await init_store()

    # Get collection info
    try:
        info = await get_collection_info()
        print(f"Collection '{info['name']}' has {info['points_count']} points")
        if info["points_count"] > 0 and not force:
            print("\nCollection already has content. Use --force to re-index.")
            return
    except Exception as e:
        print(f"Warning: Could not get collection info: {e}")

    # Find all markdown files
    if not BOOK_DOCS_PATH.exists():
        print(f"\nError: Book docs path not found: {BOOK_DOCS_PATH}")
        return

    markdown_files = sorted(BOOK_DOCS_PATH.glob("**/*.md"))
    print(f"\nFound {len(markdown_files)} chapter files to index")

    total_chunks = 0
    for chapter_path in markdown_files:
        try:
            chunks = await index_chapter(chapter_path, force)
            total_chunks += chunks
        except Exception as e:
            print(f"  Error indexing {chapter_path.name}: {e}")

    print("\n" + "=" * 60)
    print(f"Indexing complete! Total chunks indexed: {total_chunks}")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    force = "--force" in sys.argv

    asyncio.run(index_all_chapters(force))


if __name__ == "__main__":
    main()
