
from pathlib import Path

import pymupdf


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be extracted or contains no usable text."""


def extract_pdf_pages(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF while preserving one-based page numbers.

    Returns:
        A list of dictionaries containing page_number and text.
    """
    path = Path(file_path)

    if not path.is_file():
        raise PDFExtractionError(f"PDF file not found: {path}")

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise PDFExtractionError("Encrypted PDFs are not supported.")

            pages = []

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append(
                        {
                            "page_number": page_number,
                            "text": text,
                        }
                    )

    except PDFExtractionError:
        raise
    except (pymupdf.FileDataError, ValueError) as exc:
        raise PDFExtractionError("The file is not a readable PDF.") from exc

    if not pages:
        raise PDFExtractionError(
            "No extractable text found in the PDF."
        )

    return pages
