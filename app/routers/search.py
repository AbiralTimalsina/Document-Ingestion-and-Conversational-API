"""
Semantic search endpoint.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document, Chunk
from app.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.services.embedder import generate_single_embedding
from app.services import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["Search"])


@router.post("/", response_model=SearchResponse)
def semantic_search(
    request: SearchRequest,
    db: Session = Depends(get_db),
) -> SearchResponse:
    """
    Perform semantic search across all ingested documents.

    Embeds the query text, searches Qdrant for similar vectors,
    and returns matching chunks with relevance scores.
    """
    # Validate document_id filter if provided
    if request.document_id:
        doc = db.query(Document).filter(Document.id == request.document_id).first()
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{request.document_id}' not found",
            )

    # Generate query embedding
    logger.info(f"Search query: '{request.query[:100]}...'")
    query_vector = generate_single_embedding(request.query)

    # Search Qdrant
    qdrant_results = vector_store.search(
        query_vector=query_vector,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    # Enrich results with data from SQLite
    results: list[SearchResultItem] = []
    for hit in qdrant_results:
        payload = hit.get("payload", {})
        chunk_id = hit["id"]

        # Look up the chunk in SQLite for full content
        chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()

        results.append(
            SearchResultItem(
                chunk_id=chunk_id,
                document_id=payload.get("document_id", ""),
                document_filename=payload.get("document_filename", ""),
                chunk_index=payload.get("chunk_index", 0),
                content=chunk.content if chunk else payload.get("content_preview", ""),
                score=round(hit["score"], 4),
                metadata=payload,
            )
        )

    logger.info(f"Search returned {len(results)} results")

    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
    )
