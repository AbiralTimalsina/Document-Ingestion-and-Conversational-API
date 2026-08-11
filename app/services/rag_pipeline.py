import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.schemas import BookingInfo, SourceChunk
from app.services import llm, vector_store
from app.services.embedder import generate_single_embedding

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a knowledgeable assistant for a document management system. Your role is to:

1. **Answer questions** based on the provided document context. Always ground your answers in the retrieved context. If the context doesn't contain enough information, say "I don't have enough information in the documents to answer that."

2. **Help book interviews** when requested. To book an interview, you need to collect:
   - Full name
   - Email address
   - Preferred date
   - Preferred time
   Ask for each missing field naturally in conversation. When all four fields are collected, confirm the booking details.

Guidelines:
- Be conversational and helpful
- Reference the document context when answering knowledge questions
- For follow-up questions, use the conversation history to understand context
- When booking, ask for ONE missing field at a time
- Format dates as YYYY-MM-DD and times as HH:MM when confirming
- If the user provides booking information across multiple messages, remember what was already provided
- Clearly distinguish between answering document questions and booking interviews"""


QUERY_REWRITE_PROMPT = """Given the conversation history and the latest user message, rewrite the user message as a standalone search query that captures the full intent. If the message is already standalone or is about booking, return it as-is.

Conversation history:
{history}

Latest message: {message}

Rewritten search query:"""


@dataclass
class RAGResult:
    """Result of a RAG pipeline execution."""
    response: str
    sources: list[SourceChunk] = field(default_factory=list)
    booking: Optional[BookingInfo] = None
    rewritten_query: Optional[str] = None


def _rewrite_query(
    message: str,
    chat_history: list[dict[str, str]],
) -> str:
    """
    Rewrite the user's message using conversation context.

    Handles multi-turn references like "tell me more about that" or
    "what about the second point?" by incorporating prior context.

    Args:
        message: The latest user message.
        chat_history: Previous messages in OpenAI format.

    Returns:
        A standalone search query.
    """
    # Skip rewriting if no history or message is already detailed
    if not chat_history or len(message.split()) > 15:
        return message

    # Format recent history for the rewrite prompt
    recent = chat_history[-6:]  # Last 3 turns
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent
    )

    prompt = QUERY_REWRITE_PROMPT.format(history=history_text, message=message)

    rewritten = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    rewritten = rewritten.strip().strip('"')
    logger.info(f"Query rewritten: '{message}' -> '{rewritten}'")
    return rewritten


def _retrieve_context(
    query: str,
    document_id: Optional[str] = None,
) -> list[SourceChunk]:
    """
    Embed the query and retrieve relevant chunks from Qdrant.

    Args:
        query: The search query (potentially rewritten).
        document_id: Optional filter for a specific document.

    Returns:
        List of SourceChunk objects with content and scores.
    """
    settings = get_settings()

    # Generate query embedding
    query_vector = generate_single_embedding(query)

    # Search Qdrant
    results = vector_store.search(
        query_vector=query_vector,
        top_k=settings.RAG_TOP_K,
        document_id=document_id,
    )

    sources: list[SourceChunk] = []
    for hit in results:
        payload = hit.get("payload", {})
        sources.append(SourceChunk(
            chunk_id=hit["id"],
            document_id=payload.get("document_id", ""),
            document_filename=payload.get("document_filename", ""),
            content=payload.get("content_preview", ""),
            score=round(hit["score"], 4),
        ))

    logger.info(f"Retrieved {len(sources)} chunks for query")
    return sources


def _build_messages(
    chat_history: list[dict[str, str]],
    context_chunks: list[SourceChunk],
    user_message: str,
) -> list[dict[str, str]]:
    """
    Build the full messages array for the LLM call.

    Structure:
    1. System prompt with role and instructions
    2. Context block with retrieved document chunks
    3. Conversation history
    4. Current user message

    Args:
        chat_history: Previous messages in OpenAI format.
        context_chunks: Retrieved document chunks.
        user_message: The current user message.

    Returns:
        Complete messages array for the OpenAI API.
    """
    messages: list[dict[str, str]] = []

    # System prompt
    context_text = ""
    if context_chunks:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.document_filename}]\n{chunk.content}"
            )
        context_text = "\n\n".join(context_parts)

    system_content = SYSTEM_PROMPT
    if context_text:
        system_content += f"\n\n--- Retrieved Document Context ---\n{context_text}\n--- End of Context ---"

    messages.append({"role": "system", "content": system_content})

    # Conversation history
    messages.extend(chat_history)

    # Current user message
    messages.append({"role": "user", "content": user_message})

    return messages


def _detect_booking_intent(
    message: str,
    chat_history: list[dict[str, str]],
) -> bool:
    """
    Check if the conversation involves a booking intent.

    Simple heuristic check on keywords — avoids an extra LLM call.

    Args:
        message: Current user message.
        chat_history: Previous messages.

    Returns:
        True if booking intent is detected.
    """
    booking_keywords = {
        "book", "booking", "schedule", "interview", "appointment",
        "reserve", "slot", "available", "meet", "meeting",
    }

    # Check current message
    words = set(message.lower().split())
    if words & booking_keywords:
        return True

    # Check recent history for ongoing booking conversation
    for msg in chat_history[-4:]:
        content_lower = msg["content"].lower()
        if any(kw in content_lower for kw in ("booking", "interview", "appointment", "schedule")):
            return True

    return False


def _build_conversation_text(
    chat_history: list[dict[str, str]],
    current_message: str,
    ai_response: str,
) -> str:
    """Build a plaintext conversation transcript for booking extraction."""
    parts: list[str] = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")
    parts.append(f"User: {current_message}")
    parts.append(f"Assistant: {ai_response}")
    return "\n".join(parts)


def process_message(
    message: str,
    session_id: str,
    chat_history: list[dict[str, str]],
    document_id: Optional[str] = None,
) -> RAGResult:
    """
    Execute the full RAG pipeline for a user message.

    Steps:
    1. Rewrite query using conversation context (multi-turn handling)
    2. Retrieve relevant document chunks from Qdrant
    3. Build prompt with system instructions + context + history
    4. Generate response via LLM
    5. If booking intent detected, extract structured booking data

    Args:
        message: The user's current message.
        session_id: Chat session identifier.
        chat_history: Previous messages in OpenAI format.
        document_id: Optional filter for document-scoped retrieval.

    Returns:
        RAGResult with response, sources, and optional booking info.
    """
    # Step 1: Rewrite query for multi-turn context
    rewritten_query = _rewrite_query(message, chat_history)

    # Step 2: Retrieve context from Qdrant
    sources = _retrieve_context(rewritten_query, document_id)

    # Step 3: Build messages array
    messages = _build_messages(chat_history, sources, message)

    # Step 4: Generate response
    response_text = llm.chat_completion(messages, temperature=0.7)

    # Step 5: Check for booking intent and extract if present
    booking: Optional[BookingInfo] = None
    if _detect_booking_intent(message, chat_history):
        conversation_text = _build_conversation_text(
            chat_history, message, response_text
        )
        booking = llm.extract_booking(conversation_text)

    return RAGResult(
        response=response_text,
        sources=sources,
        booking=booking,
        rewritten_query=rewritten_query,
    )
