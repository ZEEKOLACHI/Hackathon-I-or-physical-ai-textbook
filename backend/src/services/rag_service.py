"""RAG service for retrieval-augmented generation."""

from typing import Any

from src.db.qdrant import search_vectors
from src.services.embedding_service import get_embedding


async def search_content(
    query: str,
    limit: int = 5,
    chapter_id: str | None = None,
    difficulty: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search textbook content using semantic search.

    Args:
        query: Search query text
        limit: Maximum number of results
        chapter_id: Optional filter by chapter
        difficulty: Optional filter by difficulty level

    Returns:
        List of search results with content and metadata
    """
    # Generate embedding for query
    query_embedding = await get_embedding(query)

    # Search Qdrant
    results = await search_vectors(
        query_vector=query_embedding,
        limit=limit,
        chapter_id=chapter_id,
        difficulty=difficulty,
    )

    return results


async def get_context_for_query(
    query: str,
    selected_text: str | None = None,
    limit: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve relevant context for a user query.

    Args:
        query: User's question
        selected_text: Optional text selected by user for context
        limit: Maximum number of context chunks

    Returns:
        Tuple of (assembled context string, list of citations)
    """
    # Combine query with selected text for better retrieval
    search_query = query
    if selected_text:
        search_query = f"{query}\n\nContext: {selected_text}"

    # Search for relevant content
    results = await search_content(search_query, limit=limit)

    if not results:
        return "", []

    # Assemble context from results
    context_parts = []
    citations = []

    for i, result in enumerate(results):
        # Build context string
        context_parts.append(
            f"[Source {i + 1}: {result.get('section_title', 'Unknown Section')} "
            f"from Chapter {result.get('chapter_id', 'Unknown')}]\n"
            f"{result.get('content', '')}\n"
        )

        # Build citation
        citations.append(
            {
                "chapter_id": result.get("chapter_id", ""),
                "section_id": result.get("section_id", ""),
                "section_title": result.get("section_title", ""),
                "relevance_score": result.get("score", 0.0),
            }
        )

    context = "\n---\n".join(context_parts)
    return context, citations


def format_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Format search results for API response.

    Args:
        results: Raw search results from Qdrant

    Returns:
        Formatted search results matching API schema
    """
    return [
        {
            "chunk_id": result.get("id", ""),
            "chapter_id": result.get("chapter_id", ""),
            "section_id": result.get("section_id", ""),
            "section_title": result.get("section_title", ""),
            "content_preview": result.get("content", "")[:200],
            "has_code": result.get("has_code", False),
            "difficulty": result.get("difficulty", "intermediate"),
            "score": result.get("score", 0.0),
        }
        for result in results
    ]
