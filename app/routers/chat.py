"""
Chat and booking endpoints for the Conversational RAG API.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Booking
from app.schemas import (
    BookingListResponse,
    BookingResponse,
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from app.services import chat_memory, rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


# POST /api/v1/chat/

@router.post("/api/v1/chat/", response_model=ChatResponse)
def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Send a message and receive a RAG-powered response.

    The pipeline retrieves relevant document chunks, incorporates
    chat history for multi-turn context, and generates a response.
    If booking intent is detected and all fields are collected,
    a booking record is created automatically.
    """
    # Load chat history from Redis
    try:
        chat_history = chat_memory.get_history_as_openai_messages(request.session_id)
    except Exception as e:
        logger.error(f"Redis error: {e}")
        raise HTTPException(status_code=503, detail="Chat memory (Redis) is unavailable")

    # Run the RAG pipeline
    try:
        result = rag_pipeline.process_message(
            message=request.message,
            session_id=request.session_id,
            chat_history=chat_history,
            document_id=request.document_id,
        )
    except RuntimeError as e:
        # Covers missing API key
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        if "AuthenticationError" in type(e).__name__ or "401" in error_msg:
            raise HTTPException(
                status_code=401,
                detail="Invalid OpenAI API key. Set OPENAI_API_KEY in your .env file.",
            )
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {error_msg}")

    # Persist messages to Redis
    chat_memory.add_message(request.session_id, "user", request.message)
    chat_memory.add_message(request.session_id, "assistant", result.response)

    # Handle booking if extracted
    booking_id: Optional[str] = None
    if result.booking:
        booking = Booking(
            session_id=request.session_id,
            name=result.booking.name,
            email=result.booking.email,
            date=result.booking.date,
            time=result.booking.time,
            status="confirmed",
        )
        db.add(booking)
        db.commit()
        db.refresh(booking)
        booking_id = booking.id
        logger.info(f"Booking created: {booking_id} for {result.booking.name}")

    return ChatResponse(
        session_id=request.session_id,
        response=result.response,
        sources=result.sources,
        booking=result.booking,
        booking_id=booking_id,
    )


# GET /api/v1/chat/{session_id}/history

@router.get("/api/v1/chat/{session_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """Retrieve the full chat history for a session from Redis."""
    messages = chat_memory.get_history(session_id)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp", ""),
            )
            for msg in messages
        ],
        total_messages=len(messages),
    )


# DELETE /api/v1/chat/{session_id}

@router.delete("/api/v1/chat/{session_id}")
def clear_chat_session(session_id: str) -> dict[str, str]:
    """Clear all chat history for a session."""
    existed = chat_memory.clear_history(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "message": f"Chat session '{session_id}' cleared",
        "session_id": session_id,
    }


# GET /api/v1/bookings/

@router.get("/api/v1/bookings/", response_model=BookingListResponse)
def list_bookings(
    session_id: Optional[str] = Query(default=None, description="Filter by session"),
    db: Session = Depends(get_db),
) -> BookingListResponse:
    """List all bookings, optionally filtered by session_id."""
    query = db.query(Booking)

    if session_id:
        query = query.filter(Booking.session_id == session_id)

    bookings = query.order_by(Booking.created_at.desc()).all()

    return BookingListResponse(
        total=len(bookings),
        bookings=[BookingResponse.model_validate(b) for b in bookings],
    )


# GET /api/v1/bookings/{booking_id}

@router.get("/api/v1/bookings/{booking_id}", response_model=BookingResponse)
def get_booking(
    booking_id: str,
    db: Session = Depends(get_db),
) -> BookingResponse:
    """Get a specific booking by ID."""
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    return BookingResponse.model_validate(booking)
