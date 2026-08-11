"""
Document management endpoints: upload, list, get, delete.
"""

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Document, Chunk
from app.schemas import (
    ChunkingStrategy,
    ChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    UploadResponse,
)
from app.services.extractor import extract_text
from app.services.chunker import chunk_text
from app.services.embedder import generate_embeddings
from app.services import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


 
# POST /upload
 

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(..., description="PDF or TXT file to upload"),
    chunking_strategy: ChunkingStrategy = Form(
        default=ChunkingStrategy.RECURSIVE,
        description="Chunking strategy: 'fixed_size' or 'recursive'",
    ),
    chunk_size: int = Form(default=None, description="Characters per chunk (optional)"),
    chunk_overlap: int = Form(default=None, description="Overlap between chunks (optional)"),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Upload a PDF or TXT file, extract text, chunk it, generate embeddings,
    and store everything in Qdrant + SQLite.
    """
    settings = get_settings()

    # ── Validate file ──────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB} MB",
        )

    # Use configured defaults if not provided
    _chunk_size = chunk_size or settings.CHUNK_SIZE
    _chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    # ── Save file to disk ──────────────────────
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Unique filename to avoid collisions
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}{ext}"
    file_path = upload_dir / safe_filename

    with open(file_path, "wb") as f:
        f.write(content)

    #  Create document record
    file_type = ext.lstrip(".")
    doc = Document(
        id=file_id,
        filename=file.filename,
        file_type=file_type,
        file_size=file_size,
        chunking_strategy=chunking_strategy.value,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        # Extract text 
        logger.info(f"Extracting text from '{file.filename}'...")
        raw_text = extract_text(str(file_path), file_type)

        #  Chunk text 
        logger.info(f"Chunking with strategy='{chunking_strategy.value}'...")
        chunks = chunk_text(
            raw_text,
            strategy=chunking_strategy.value,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
        )

        if not chunks:
            raise ValueError("Text extraction produced no chunks")

        #  Generate embeddings 
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [c.content for c in chunks]
        embeddings = generate_embeddings(texts)

        # Store in Qdrant
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        payloads = [
            {
                "document_id": doc.id,
                "document_filename": file.filename,
                "chunk_index": c.index,
                "content_preview": c.content[:200],
                "char_count": c.char_count,
            }
            for c in chunks
        ]
        vector_store.upsert_vectors(embeddings, chunk_ids, payloads)

        # Save chunks to SQLite 
        db_chunks = []
        for chunk_result, chunk_id in zip(chunks, chunk_ids):
            db_chunk = Chunk(
                id=chunk_id,
                document_id=doc.id,
                chunk_index=chunk_result.index,
                content=chunk_result.content,
                char_count=chunk_result.char_count,
                embedding_id=chunk_id,
                metadata_json={
                    "chunking_strategy": chunking_strategy.value,
                    "chunk_size": _chunk_size,
                    "chunk_overlap": _chunk_overlap,
                },
            )
            db_chunks.append(db_chunk)

        db.add_all(db_chunks)

        # Update document status
        doc.status = "completed"
        doc.total_chunks = len(chunks)
        db.commit()
        db.refresh(doc)

        logger.info(
            f"Document '{file.filename}' processed successfully: "
            f"{len(chunks)} chunks created"
        )

        return UploadResponse(
            message=f"Document uploaded and processed successfully. {len(chunks)} chunks created.",
            document=DocumentResponse.model_validate(doc),
        )

    except Exception as e:
        # Mark document as failed
        doc.status = "failed"
        doc.error_message = str(e)
        db.commit()
        logger.error(f"Failed to process '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


 
# GET / (list documents)
 

@router.get("/", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """List all uploaded documents with pagination."""
    total = db.query(Document).count()
    offset = (page - 1) * page_size

    documents = (
        db.query(Document)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        documents=[DocumentResponse.model_validate(d) for d in documents],
    )


 
# GET /{document_id}
 

@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDetailResponse:
    """Get a single document with all its chunks."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetailResponse.model_validate(doc)


 
# GET /{document_id}/chunks
 

@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
def get_document_chunks(
    document_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[ChunkResponse]:
    """List chunks for a specific document with pagination."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    offset = (page - 1) * page_size
    chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return [ChunkResponse.model_validate(c) for c in chunks]


 
# DELETE /{document_id}
 

@router.delete("/{document_id}", status_code=200)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    """
    Delete a document, its chunks from SQLite, and its vectors from Qdrant.
    Also removes the uploaded file from disk.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors from Qdrant
    try:
        vector_store.delete_by_document(document_id)
    except Exception as e:
        logger.warning(f"Failed to delete vectors from Qdrant: {e}")

    # Remove uploaded file from disk
    settings = get_settings()
    file_path = Path(settings.UPLOAD_DIR) / f"{document_id}.{doc.file_type}"
    if file_path.exists():
        os.remove(file_path)

    # Delete from SQLite (cascade deletes chunks)
    db.delete(doc)
    db.commit()

    return {
        "message": f"Document '{doc.filename}' and its {doc.total_chunks} chunks deleted successfully",
        "document_id": document_id,
    }
