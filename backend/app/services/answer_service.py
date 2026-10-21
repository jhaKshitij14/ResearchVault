import json
import re
from typing import Any


ABSTENTION_MESSAGE = (
    "I could not find sufficient evidence in the retrieved passages "
    "to answer this question."
)


def build_answer_prompt(
    question: str,
    passages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Build a prompt with stable citation IDs and inspectable source metadata."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")

    sources = []
    evidence_blocks = []

    for index, passage in enumerate(passages, start=1):
        citation_id = f"S{index}"
        source = {
            "citation_id": citation_id,
            "chunk_id": passage["chunk_id"],
            "document_id": passage["document_id"],
            "chunk_index": passage["chunk_index"],
            "page_number": passage["page_number"],
            "text": passage["text"],
            "distance": passage["distance"],
        }
        sources.append(source)
        evidence_blocks.append(
            f"[{citation_id}]\n"
            f"Document: {passage['document_id']}\n"
            f"Page: {passage['page_number']}\n"
            f"Passage: {passage['text']}"
        )

    evidence = "\n\n".join(evidence_blocks)

    prompt = f"""
You are ResearchVault, an evidence-first technical knowledge assistant.

Answer the question using only the evidence passages supplied below.
Do not invent facts or use outside knowledge.
If the passages do not sufficiently support an answer, abstain.
For every non-abstaining answer, include inline citations such as [S1].
Only cite IDs that appear in the supplied evidence.
Return only a valid JSON object with exactly these fields:
- "answer": string
- "abstained": boolean
- "citations": array of citation ID strings

Question:
{question}

Evidence passages:
{evidence}
""".strip()

    return prompt, sources


async def generate_grounded_answer(
    question: str,
    passages: list[dict[str, Any]],
    llm_client: Any,
) -> dict[str, Any]:
    """Generate an evidence-grounded answer and validate its citations."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")

    if not passages:
        return {
            "question": question,
            "answer": ABSTENTION_MESSAGE,
            "abstained": True,
            "citations": [],
            "sources": [],
        }

    prompt, sources = build_answer_prompt(question, passages)
    raw_response = await llm_client.generate(prompt)

    try:
        result = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("LLM returned invalid JSON") from exc

    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")

    answer = result.get("answer")
    abstained = result.get("abstained")
    citations = result.get("citations")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM answer must be a non-empty string")
    if not isinstance(abstained, bool):
        raise ValueError("LLM abstained field must be a boolean")
    if not isinstance(citations, list) or any(
        not isinstance(citation, str) for citation in citations
    ):
        raise ValueError("LLM citations must be a list of strings")

    if abstained:
        return {
            "question": question,
            "answer": answer.strip(),
            "abstained": True,
            "citations": [],
            "sources": [],
        }

    valid_ids = {source["citation_id"] for source in sources}
    if (
        len(citations) != len(set(citations))
        or any(citation not in valid_ids for citation in citations)
    ):
        raise ValueError("LLM returned unknown or duplicate citations")

    if not citations:
        raise ValueError("A non-abstaining answer must cite at least one source")

    # Require every citation listed in JSON to appear in the answer text.
    inline_ids = set(re.findall(r"\[(S\d+)\]", answer))
    if not set(citations).issubset(inline_ids):
        raise ValueError("A non-abstaining answer must cite its sources inline")

    source_by_id = {source["citation_id"]: source for source in sources}

    return {
        "question": question,
        "answer": answer.strip(),
        "abstained": False,
        "citations": citations,
        "sources": [source_by_id[citation] for citation in citations],
    }
