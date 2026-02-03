"""Local JSON + numpy vector store replacing Qdrant."""

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# Storage configuration
STORE_PATH = Path(__file__).parent.parent.parent / "vector_store.json"
VECTOR_SIZE = 768  # Google text-embedding-004 dimension

# In-memory store
_vectors: list[dict[str, Any]] = []
_dirty: bool = False
_loaded: bool = False


def _load() -> None:
    """Load vectors from JSON file into memory."""
    global _vectors, _loaded
    if _loaded:
        return
    if STORE_PATH.exists():
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        _vectors = data.get("vectors", [])
    else:
        _vectors = []
    _loaded = True


def _ensure_loaded() -> None:
    """Ensure vectors are loaded (lazy initialization for serverless)."""
    if not _loaded:
        _load()


def _save() -> None:
    """Persist vectors to JSON file."""
    global _dirty
    STORE_PATH.write_text(
        json.dumps({"vectors": _vectors}, ensure_ascii=False),
        encoding="utf-8",
    )
    _dirty = False


async def init_store() -> None:
    """Initialize vector store (load from disk)."""
    _load()
    print(f"Vector store loaded: {len(_vectors)} vectors from {STORE_PATH}")


async def close_store() -> None:
    """Persist and close vector store."""
    global _dirty
    if _dirty:
        _save()


async def upsert_vectors(
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    ids: list[str] | None = None,
) -> None:
    """
    Upsert vectors with payloads into the local store.

    Args:
        vectors: List of embedding vectors
        payloads: List of metadata payloads for each vector
        ids: Optional list of IDs (generates UUIDs if not provided)
    """
    global _dirty

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in vectors]

    # Build a set of existing IDs for fast lookup
    existing_ids = {v["id"]: i for i, v in enumerate(_vectors)}

    for id_, vector, payload in zip(ids, vectors, payloads):
        entry = {"id": id_, "vector": vector, "payload": payload}
        if id_ in existing_ids:
            _vectors[existing_ids[id_]] = entry
        else:
            _vectors.append(entry)

    _dirty = True
    _save()  # persist immediately


async def search_vectors(
    query_vector: list[float],
    limit: int = 5,
    chapter_id: str | None = None,
    difficulty: str | None = None,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Search for similar vectors using cosine similarity.

    Args:
        query_vector: Query embedding vector
        limit: Maximum number of results
        chapter_id: Optional filter by chapter
        difficulty: Optional filter by difficulty level
        score_threshold: Minimum similarity score

    Returns:
        List of search results with payloads and scores
    """
    _ensure_loaded()
    if not _vectors:
        return []

    # Filter candidates first
    candidates = _vectors
    if chapter_id:
        candidates = [v for v in candidates if v["payload"].get("chapter_id") == chapter_id]
    if difficulty:
        candidates = [v for v in candidates if v["payload"].get("difficulty") == difficulty]

    if not candidates:
        return []

    # Build matrix of candidate vectors
    mat = np.array([c["vector"] for c in candidates], dtype=np.float32)
    query = np.array(query_vector, dtype=np.float32)

    # Cosine similarity
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []
    mat_norms = np.linalg.norm(mat, axis=1)
    mat_norms[mat_norms == 0] = 1e-10
    scores = mat @ query / (mat_norms * query_norm)

    # Filter by threshold and sort descending
    indices = np.where(scores >= score_threshold)[0]
    if len(indices) == 0:
        return []

    sorted_idx = indices[np.argsort(-scores[indices])][:limit]

    return [
        {
            "id": candidates[i]["id"],
            "score": float(scores[i]),
            **candidates[i]["payload"],
        }
        for i in sorted_idx
    ]


async def delete_by_chapter(chapter_id: str) -> None:
    """Delete all vectors for a specific chapter."""
    global _vectors, _dirty
    _vectors = [v for v in _vectors if v["payload"].get("chapter_id") != chapter_id]
    _dirty = True
    _save()


async def get_collection_info() -> dict[str, Any]:
    """Get collection statistics."""
    _ensure_loaded()
    return {
        "name": "textbook_content",
        "vectors_count": len(_vectors),
        "points_count": len(_vectors),
    }
