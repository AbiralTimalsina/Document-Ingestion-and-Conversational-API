"""
Text extraction service for PDF and TXT files.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Concatenated text from all pages, separated by newlines.

    Raises:
        ValueError: If the file cannot be read or contains no extractable text.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    try:
        doc = fitz.open(str(path))
        pages_text = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
            logger.debug(f"Page {page_num}: extracted {len(text)} characters")

        doc.close()

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            raise ValueError(
                f"No extractable text found in '{path.name}'. "
                "The PDF may be image-based (scanned). OCR is not supported yet."
            )

        logger.info(
            f"Extracted {len(full_text)} characters from '{path.name}' "
            f"({len(pages_text)} pages)"
        )
        return full_text

    except fitz.fitz.FileDataError as e:
        raise ValueError(f"Failed to read PDF '{path.name}': {e}")


def extract_text_from_txt(file_path: str) -> str:
    """
    Read all text from a plain text file with encoding detection fallback.

    Args:
        file_path: Absolute or relative path to the TXT file.

    Returns:
        The full text content of the file.

    Raises:
        ValueError: If the file cannot be read.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    
    for encoding in ("utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            logger.info(
                f"Read {len(text)} characters from '{path.name}' "
                f"(encoding={encoding})"
            )
            return text
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Failed to decode text file '{path.name}'")


def extract_text(file_path: str, file_type: str) -> str:
    """
    Route to the appropriate extractor based on file type.

    Args:
        file_path: Path to the uploaded file.
        file_type: Either "pdf" or "txt".

    Returns:
        Extracted text content.
    """
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
