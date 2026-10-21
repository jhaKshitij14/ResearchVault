
import pytest

from app.services.chunker import chunk_pages


def test_splits_text_into_chunks():
    words = [f"word{i}" for i in range(1100)]
    pages = [{"page_number": 1, "text": " ".join(words)}]

    chunks = chunk_pages(pages)

    assert len(chunks) == 3
    assert len(chunks[0]["text"].split()) == 500
    assert len(chunks[1]["text"].split()) == 500
    assert len(chunks[2]["text"].split()) == 250


def test_chunks_have_expected_overlap():
    words = [f"word{i}" for i in range(600)]
    pages = [{"page_number": 1, "text": " ".join(words)}]

    chunks = chunk_pages(pages)

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()

    assert first_words[-75:] == second_words[:75]


def test_preserves_original_page_numbers():
    pages = [
        {"page_number": 2, "text": " ".join(["alpha"] * 20)},
        {"page_number": 5, "text": " ".join(["beta"] * 20)},
    ]

    chunks = chunk_pages(pages)

    assert [chunk["page_number"] for chunk in chunks] == [2, 5]


def test_empty_page_produces_no_chunks():
    pages = [{"page_number": 3, "text": ""}]

    assert chunk_pages(pages) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_rejects_invalid_settings(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(
            [{"page_number": 1, "text": "some text"}],
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunk_indices_are_sequential():
    pages = [
        {"page_number": 1, "text": " ".join(["alpha"] * 600)},
        {"page_number": 2, "text": " ".join(["beta"] * 600)},
    ]

    chunks = chunk_pages(pages)

    assert [chunk["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )

