import json
import logging
from typing import Optional

from openai import OpenAI

from app.config import get_settings
from app.schemas import BookingInfo

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Get or create the OpenAI client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"OpenAI client initialized (model: {settings.OPENAI_MODEL})")
    return _client


def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
) -> str:
    """
    Call OpenAI chat completions API.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        The assistant's response text.
    """
    settings = get_settings()
    client = get_client()

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )

    content = response.choices[0].message.content or ""
    logger.info(
        f"LLM response: {len(content)} chars, "
        f"tokens: {response.usage.prompt_tokens}+{response.usage.completion_tokens}"
    )
    return content


BOOKING_EXTRACTION_PROMPT = """Analyze the conversation below and extract interview booking information.
If ALL four fields (name, email, date, time) are present in the conversation, return them as JSON.
If any field is missing or unclear, return exactly: null

Return ONLY valid JSON in this exact format (or null):
{
    "name": "full name",
    "email": "email@example.com",
    "date": "YYYY-MM-DD",
    "time": "HH:MM"
}

Conversation:
"""


def extract_booking(conversation_text: str) -> Optional[BookingInfo]:
    """
    Use the LLM to extract structured booking data from a conversation.

    Uses a focused prompt to pull out name, email, date, and time.
    Returns None if any field is missing.

    Args:
        conversation_text: The full conversation text to analyze.

    Returns:
        BookingInfo if all fields were extracted, None otherwise.
    """
    settings = get_settings()
    client = get_client()

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract structured data from conversations. Return only JSON or null.",
            },
            {
                "role": "user",
                "content": BOOKING_EXTRACTION_PROMPT + conversation_text,
            },
        ],
        temperature=0.0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content or ""
    raw = raw.strip()

    # Handle "null" response
    if raw.lower() == "null" or not raw:
        logger.debug("Booking extraction returned null (incomplete data)")
        return None

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
        booking = BookingInfo(**data)
        logger.info(f"Extracted booking: {booking.name}, {booking.email}")
        return booking
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse booking extraction: {e}, raw='{raw}'")
        return None
