"""
Embedding generation service using sentence-transformers.

Uses the all-MiniLM-L6-v2 model by default (384 dimensions).
The model is loaded once and cached as a module-level singleton.
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once at startup
_model: Optional[SentenceTransformer] = None


def load_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model (downloads on first run).

    Returns:
        The loaded SentenceTransformer model.
    """
    global _model
    if _model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(
            f"Model loaded. Dimension: {_model.get_embedding_dimension()}"
        )
    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = load_model()

    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=32,
    )

    # Convert numpy arrays to plain Python lists for JSON serialization
    result = [emb.tolist() for emb in embeddings]
    logger.info(f"Generated {len(result)} embeddings (dim={len(result[0])})")
    return result


def generate_single_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text string.

    Args:
        text: The text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    results = generate_embeddings([text])
    return results[0]
