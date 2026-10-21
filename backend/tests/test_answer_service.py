import json
from unittest.mock import AsyncMock

import pytest

from app.services.answer_service import (
    ABSTENTION_MESSAGE,
    build_answer_prompt,
    generate_grounded_answer,
)


def make_passage(citation_text="PostgreSQL supports vector search."):
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "chunk_index": 0,
        "page_number": 3,
        "text": citation_text,
        "distance": 0.1,
    }


@pytest.mark.anyio
async def test_returns_answer_with_inspectable_cited_source():
    client = AsyncMock()
    client.generate.return_value = json.dumps({
        "answer": "PostgreSQL supports vector search [S1].",
        "abstained": False,
        "citations": ["S1"],
    })

    result = await generate_grounded_answer(
        "What does PostgreSQL support?",
        [make_passage()],
        client,
    )

    assert result["abstained"] is False
    assert result["citations"] == ["S1"]
    assert result["sources"][0]["page_number"] == 3
    assert result["sources"][0]["text"] == "PostgreSQL supports vector search."
    client.generate.assert_awaited_once()


@pytest.mark.anyio
async def test_abstains_without_calling_llm_when_no_passages():
    client = AsyncMock()

    result = await generate_grounded_answer(
        "What does this document say?",
        [],
        client,
    )

    assert result["abstained"] is True
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["sources"] == []
    client.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_accepts_model_abstention():
    client = AsyncMock()
    client.generate.return_value = json.dumps({
        "answer": "The available passages do not establish this.",
        "abstained": True,
        "citations": [],
    })

    result = await generate_grounded_answer(
        "What is not covered by the evidence?",
        [make_passage()],
        client,
    )

    assert result["abstained"] is True
    assert result["citations"] == []
    assert result["sources"] == []


@pytest.mark.anyio
async def test_rejects_unknown_citation():
    client = AsyncMock()
    client.generate.return_value = json.dumps({
        "answer": "Unsupported claim [S9].",
        "abstained": False,
        "citations": ["S9"],
    })

    with pytest.raises(ValueError, match="unknown or duplicate"):
        await generate_grounded_answer(
            "Question?",
            [make_passage()],
            client,
        )


@pytest.mark.anyio
async def test_rejects_answer_without_citations():
    client = AsyncMock()
    client.generate.return_value = json.dumps({
        "answer": "An answer without evidence.",
        "abstained": False,
        "citations": [],
    })

    with pytest.raises(ValueError, match="must cite"):
        await generate_grounded_answer(
            "Question?",
            [make_passage()],
            client,
        )


@pytest.mark.anyio
async def test_rejects_invalid_json():
    client = AsyncMock()
    client.generate.return_value = "This is not JSON"

    with pytest.raises(ValueError, match="invalid JSON"):
        await generate_grounded_answer(
            "Question?",
            [make_passage()],
            client,
        )


def test_prompt_includes_question_evidence_and_page_provenance():
    prompt, sources = build_answer_prompt(
        "What does PostgreSQL support?",
        [make_passage()],
    )

    assert "What does PostgreSQL support?" in prompt
    assert "[S1]" in prompt
    assert "Page: 3" in prompt
    assert sources[0]["citation_id"] == "S1"


@pytest.mark.anyio
async def test_rejects_empty_question():
    client = AsyncMock()

    with pytest.raises(ValueError, match="Question must not be empty"):
        await generate_grounded_answer("  ", [make_passage()], client)

    client.generate.assert_not_awaited()
