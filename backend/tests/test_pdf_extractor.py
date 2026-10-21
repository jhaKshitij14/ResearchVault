
import pymupdf
import pytest

from app.services.pdf_extractor import (
    PDFExtractionError,
    extract_pdf_pages,
)


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page1 = document.new_page()
        page1.insert_text((72, 72), "ResearchVault introduction")

        document.new_page()  # Intentionally blank page

        page3 = document.new_page()
        page3.insert_text((72, 72), "ResearchVault conclusion")

        document.save(pdf_path)

    return pdf_path


def test_extracts_text_with_original_page_numbers(sample_pdf):
    pages = extract_pdf_pages(sample_pdf)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "ResearchVault introduction" in pages[0]["text"]
    assert pages[1]["page_number"] == 3
    assert "ResearchVault conclusion" in pages[1]["text"]


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(PDFExtractionError, match="not found"):
        extract_pdf_pages(tmp_path / "missing.pdf")


def test_invalid_pdf_raises_error(tmp_path):
    invalid_path = tmp_path / "invalid.pdf"
    invalid_path.write_text("This is not a PDF")

    with pytest.raises(PDFExtractionError, match="not a readable PDF"):
        extract_pdf_pages(invalid_path)


def test_pdf_without_extractable_text_raises_error(tmp_path):
    pdf_path = tmp_path / "blank.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    with pytest.raises(PDFExtractionError, match="No extractable text"):
        extract_pdf_pages(pdf_path)
