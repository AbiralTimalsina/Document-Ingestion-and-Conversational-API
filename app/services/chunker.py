"""
Text chunking service with two selectable strategies.

Strategies:
    1. fixed_size  — Split text into chunks of N characters with M overlap.
    2. recursive   — Recursively split by paragraph → newline → sentence → space
                     boundaries, keeping chunks under the size limit.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Represents a single text chunk."""

    index: int
    content: str
    char_count: int


 
# Strategy 1: Fixed-size chunking
 

def chunk_fixed_size(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[ChunkResult]:
    """
    Split text into fixed-size character chunks with overlap.

    This is the simplest strategy — predictable, fast, but may cut words
    or sentences in the middle.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of ChunkResult objects.
    """
    if not text.strip():
        return []

    chunks: list[ChunkResult] = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                ChunkResult(
                    index=index,
                    content=chunk_text,
                    char_count=len(chunk_text),
                )
            )
            index += 1

        # Move forward by (chunk_size - overlap)
        start += chunk_size - chunk_overlap

    logger.info(
        f"Fixed-size chunking: {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


 
# Strategy 2: Recursive character chunking
 

# Separators ordered from coarsest to finest grain
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """
    Recursively split text using progressively finer separators.

    Tries the coarsest separator first (paragraph break). If any resulting
    segment is still too large, it recurses with the next finer separator.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best (coarsest) separator present in the text
    separator = separators[-1]  # fallback to space
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    # Split with the chosen separator
    parts = text.split(separator)

    # Merge parts into chunks that fit within chunk_size
    chunks: list[str] = []
    current_chunk = ""

    for part in parts:
        # Add separator back (except for the first piece)
        candidate = (
            current_chunk + separator + part if current_chunk else part
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Save the current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # If this single part exceeds chunk_size, recurse deeper
            if len(part) > chunk_size and remaining_separators:
                sub_chunks = _split_text_recursive(
                    part, chunk_size, chunk_overlap, remaining_separators
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap by prepending the tail of the previous chunk."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        merged = prev_tail + " " + chunks[i]
        result.append(merged.strip())
    return result


def chunk_recursive(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[ChunkResult]:
    """
    Recursively split text respecting paragraph and sentence boundaries.

    This strategy preserves semantic coherence better than fixed-size chunking
    by splitting at natural text boundaries (\n\n → \n → ". " → " ").

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of ChunkResult objects.
    """
    if not text.strip():
        return []

    raw_chunks = _split_text_recursive(text, chunk_size, chunk_overlap, _SEPARATORS)
    raw_chunks = _add_overlap(raw_chunks, chunk_overlap)

    chunks = [
        ChunkResult(index=i, content=c, char_count=len(c))
        for i, c in enumerate(raw_chunks)
        if c.strip()
    ]

    logger.info(
        f"Recursive chunking: {len(chunks)} chunks "
        f"(target_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


 
# Public dispatcher
 

def chunk_text(
    text: str,
    strategy: str = "fixed_size",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[ChunkResult]:
    """
    Chunk text using the selected strategy.

    Args:
        text: The input text.
        strategy: Either "fixed_size" or "recursive".
        chunk_size: Max characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of ChunkResult objects.

    Raises:
        ValueError: If the strategy is unknown.
    """
    if strategy == "fixed_size":
        return chunk_fixed_size(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return chunk_recursive(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Choose 'fixed_size' or 'recursive'."
        )
