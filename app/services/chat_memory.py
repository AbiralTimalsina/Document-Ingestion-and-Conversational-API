import json
import logging
from datetime import datetime, timezone
from typing import Optional

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """Get or create the Redis client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        logger.info(f"Redis client connected to {settings.REDIS_URL}")
    return _client


def ping() -> bool:
    """Check if Redis is reachable."""
    try:
        client = get_redis_client()
        return client.ping()
    except redis.ConnectionError:
        return False


def _session_key(session_id: str) -> str:
    """Build the Redis key for a chat session."""
    return f"chat:{session_id}"


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Append a message to a session's chat history.

    Args:
        session_id: Unique session identifier.
        role: Message role ("user" or "assistant").
        content: Message text content.
    """
    settings = get_settings()
    client = get_redis_client()
    key = _session_key(session_id)

    message = json.dumps({
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    client.rpush(key, message)
    client.expire(key, settings.CHAT_HISTORY_TTL)

    # Trim to max history length
    client.ltrim(key, -settings.MAX_HISTORY_MESSAGES, -1)

    logger.debug(f"Added {role} message to session '{session_id}'")


def get_history(session_id: str) -> list[dict[str, str]]:
    """
    Retrieve full chat history for a session.

    Args:
        session_id: Unique session identifier.

    Returns:
        List of message dicts with keys: role, content, timestamp.
    """
    client = get_redis_client()
    key = _session_key(session_id)

    raw_messages = client.lrange(key, 0, -1)
    messages: list[dict[str, str]] = [json.loads(msg) for msg in raw_messages]

    logger.debug(f"Retrieved {len(messages)} messages for session '{session_id}'")
    return messages


def get_history_as_openai_messages(session_id: str) -> list[dict[str, str]]:
    """
    Retrieve chat history formatted for the OpenAI messages array.

    Returns only role and content fields (no timestamp).

    Args:
        session_id: Unique session identifier.

    Returns:
        List of dicts with keys: role, content.
    """
    history = get_history(session_id)
    return [{"role": msg["role"], "content": msg["content"]} for msg in history]


def clear_history(session_id: str) -> bool:
    """
    Delete all chat history for a session.

    Args:
        session_id: Unique session identifier.

    Returns:
        True if the session existed and was deleted.
    """
    client = get_redis_client()
    key = _session_key(session_id)
    deleted = client.delete(key)
    logger.info(f"Cleared chat history for session '{session_id}' (existed={bool(deleted)})")
    return bool(deleted)


def get_message_count(session_id: str) -> int:
    """Get the number of messages in a session."""
    client = get_redis_client()
    key = _session_key(session_id)
    return client.llen(key)
