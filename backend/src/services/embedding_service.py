"""Embedding service using Google AI text-embedding-004 model."""

import google.generativeai as genai

from src.config import settings

# Initialize Google AI
_initialized = False

# Embedding model configuration
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSIONS = 768  # Google's embedding dimension


def _init_google_ai() -> None:
    """Initialize Google AI client."""
    global _initialized
    if not _initialized:
        genai.configure(api_key=settings.google_api_key)
        _initialized = True


async def get_embedding(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    _init_google_ai()
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    _init_google_ai()
    embeddings = []

    # Google AI supports batch embedding
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])

    return embeddings


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to find a natural break point (paragraph or sentence)
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                sentence_break = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                )
                if sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks
