"""Qdrant vector database client and collection management."""

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import settings

# Collection configuration
COLLECTION_NAME = "textbook_content"
VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small dimension

# Initialize Qdrant client
qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global qdrant_client
    if qdrant_client is None:
        # Use local storage if URL is "local" or not set
        if not settings.qdrant_url or settings.qdrant_url.lower() == "local":
            # Use local file-based storage (persists data)
            qdrant_client = QdrantClient(path="./qdrant_data")
        else:
            qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
    return qdrant_client


async def init_qdrant() -> None:
    """Initialize Qdrant collection if it doesn't exist."""
    client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


async def close_qdrant() -> None:
    """Close Qdrant client connection."""
    global qdrant_client
    if qdrant_client is not None:
        qdrant_client.close()
        qdrant_client = None


def get_collection_name() -> str:
    """Get the textbook content collection name."""
    return COLLECTION_NAME


async def upsert_vectors(
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    ids: list[str] | None = None,
) -> None:
    """
    Upsert vectors with payloads into Qdrant.

    Args:
        vectors: List of embedding vectors
        payloads: List of metadata payloads for each vector
        ids: Optional list of IDs (generates UUIDs if not provided)
    """
    client = get_qdrant_client()

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in vectors]

    points = [
        PointStruct(id=id_, vector=vector, payload=payload)
        for id_, vector, payload in zip(ids, vectors, payloads)
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


async def search_vectors(
    query_vector: list[float],
    limit: int = 5,
    chapter_id: str | None = None,
    difficulty: str | None = None,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Search for similar vectors in Qdrant.

    Args:
        query_vector: Query embedding vector
        limit: Maximum number of results
        chapter_id: Optional filter by chapter
        difficulty: Optional filter by difficulty level
        score_threshold: Minimum similarity score

    Returns:
        List of search results with payloads and scores
    """
    client = get_qdrant_client()

    # Build filter conditions
    conditions = []
    if chapter_id:
        conditions.append(
            FieldCondition(key="chapter_id", match=MatchValue(value=chapter_id))
        )
    if difficulty:
        conditions.append(
            FieldCondition(key="difficulty", match=MatchValue(value=difficulty))
        )

    query_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        query_filter=query_filter,
        score_threshold=score_threshold,
    )

    return [
        {
            "id": str(result.id),
            "score": result.score,
            **result.payload,
        }
        for result in results
    ]


async def delete_by_chapter(chapter_id: str) -> None:
    """Delete all vectors for a specific chapter."""
    client = get_qdrant_client()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="chapter_id", match=MatchValue(value=chapter_id))]
        ),
    )


async def get_collection_info() -> dict[str, Any]:
    """Get collection statistics."""
    client = get_qdrant_client()
    info = client.get_collection(COLLECTION_NAME)
    return {
        "name": COLLECTION_NAME,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
    }
