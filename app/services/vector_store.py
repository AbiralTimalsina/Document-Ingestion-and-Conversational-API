"""
Qdrant vector store service.

Uses Qdrant in local on-disk mode — no Docker or cloud account needed.
Data persists to the QDRANT_PATH directory.
"""

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton
_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """
    Get or create the Qdrant client in local on-disk mode.

    Returns:
        An initialized QdrantClient pointing to the local storage path.
    """
    global _client
    if _client is None:
        settings = get_settings()
        logger.info(f"Initializing Qdrant client at: {settings.QDRANT_PATH}")
        _client = QdrantClient(path=settings.QDRANT_PATH)
        logger.info("Qdrant client initialized (local on-disk mode)")
    return _client


def ensure_collection() -> None:
    """
    Create the vector collection if it doesn't already exist.

    Uses cosine distance and the configured embedding dimension (default: 384).
    """
    settings = get_settings()
    client = get_client()

    collections = [c.name for c in client.get_collections().collections]

    if settings.COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            f"Created Qdrant collection '{settings.COLLECTION_NAME}' "
            f"(dim={settings.EMBEDDING_DIMENSION}, distance=cosine)"
        )
    else:
        logger.info(f"Qdrant collection '{settings.COLLECTION_NAME}' already exists")


def upsert_vectors(
    embeddings: list[list[float]],
    chunk_ids: list[str],
    payloads: list[dict],
) -> list[str]:
    """
    Upsert embedding vectors into Qdrant with associated metadata payloads.

    Args:
        embeddings: List of embedding vectors.
        chunk_ids: List of chunk UUIDs to use as Qdrant point IDs.
        payloads: List of metadata dicts (document_id, chunk_index, content_preview, etc.)

    Returns:
        List of Qdrant point IDs (same as chunk_ids).
    """
    settings = get_settings()
    client = get_client()

    points = [
        PointStruct(
            id=cid,
            vector=embedding,
            payload=payload,
        )
        for cid, embedding, payload in zip(chunk_ids, embeddings, payloads)
    ]

    # Batch upsert in groups of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=batch,
        )

    logger.info(f"Upserted {len(points)} vectors into Qdrant")
    return chunk_ids


def search(
    query_vector: list[float],
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> list[dict]:
    """
    Perform similarity search in Qdrant.

    Args:
        query_vector: The query embedding vector.
        top_k: Number of results to return.
        document_id: Optional filter to limit results to a specific document.

    Returns:
        List of dicts with keys: id, score, payload.
    """
    settings = get_settings()
    client = get_client()

    # Build optional filter
    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results.points
    ]


def delete_by_document(document_id: str) -> None:
    """
    Delete all vectors associated with a document.

    Args:
        document_id: The document ID whose vectors should be removed.
    """
    settings = get_settings()
    client = get_client()

    client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )
    logger.info(f"Deleted vectors for document {document_id} from Qdrant")


def get_collection_info() -> dict:
    """Get information about the vector collection (point count, etc.)."""
    settings = get_settings()

    try:
        client = get_client()
        info = client.get_collection(settings.COLLECTION_NAME)
        return {
            "collection": settings.COLLECTION_NAME,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {
            "collection": settings.COLLECTION_NAME,
            "status": "not_initialized",
            "detail": str(e),
        }
