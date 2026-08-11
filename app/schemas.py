

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


 
# Enums
 

class ChunkingStrategy(str, Enum):
    """Available text chunking strategies."""
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


 
# Response schemas
 

class ChunkResponse(BaseModel):
    """Schema for a single chunk in API responses."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    char_count: int
    embedding_id: Optional[str] = None
    metadata_json: Optional[dict] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    """Schema for document metadata in API responses."""
    id: str
    filename: str
    file_type: str
    file_size: int
    total_chunks: int
    chunking_strategy: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentDetailResponse(DocumentResponse):
    """Document response including its chunks."""
    chunks: list[ChunkResponse] = []


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    total: int
    page: int
    page_size: int
    documents: list[DocumentResponse]


class UploadResponse(BaseModel):
    """Response after a successful file upload."""
    message: str
    document: DocumentResponse


 
# Search schemas
 

class SearchRequest(BaseModel):
    """Semantic search request body."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    document_id: Optional[str] = Field(
        default=None, description="Optional: limit search to a specific document"
    )


class SearchResultItem(BaseModel):
    """A single search result."""
    chunk_id: str
    document_id: str
    document_filename: str
    chunk_index: int
    content: str
    score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    """Search results response."""
    query: str
    results: list[SearchResultItem]
    total_results: int


 
# Health check
 

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: dict


# Chat schemas

class ChatRequest(BaseModel):
    """Conversational RAG request body."""
    session_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Unique session identifier for conversation continuity",
    )
    message: str = Field(
        ..., min_length=1, max_length=5000,
        description="User message",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional: limit retrieval to a specific document",
    )


class SourceChunk(BaseModel):
    """A retrieved chunk used as context for the response."""
    chunk_id: str
    document_id: str
    document_filename: str
    content: str
    score: float


class BookingInfo(BaseModel):
    """Structured booking data extracted by the LLM."""
    name: str
    email: str
    date: str
    time: str


class ChatResponse(BaseModel):
    """Conversational RAG response."""
    session_id: str
    response: str
    sources: list[SourceChunk] = []
    booking: Optional[BookingInfo] = None
    booking_id: Optional[str] = None


class ChatMessage(BaseModel):
    """A single message in chat history."""
    role: str
    content: str
    timestamp: str


class ChatHistoryResponse(BaseModel):
    """Chat history for a session."""
    session_id: str
    messages: list[ChatMessage]
    total_messages: int


class BookingResponse(BaseModel):
    """Booking record response."""
    id: str
    session_id: str
    name: str
    email: str
    date: str
    time: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class BookingListResponse(BaseModel):
    """List of bookings."""
    total: int
    bookings: list[BookingResponse]
