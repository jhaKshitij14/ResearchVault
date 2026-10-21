
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 75,
) -> list[dict]:
    """
    Split page text into overlapping word-based chunks.

    Each input page must contain:
        page_number: int
        text: str

    Each output chunk contains:
        chunk_index: int
        page_number: int
        text: str
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    chunk_index = 0
    step = chunk_size - overlap

    for page in pages:
        page_number = page["page_number"]
        words = page["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                break

            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "text": " ".join(chunk_words),
                }
            )
            chunk_index += 1

            if start + chunk_size >= len(words):
                break

    return chunks
