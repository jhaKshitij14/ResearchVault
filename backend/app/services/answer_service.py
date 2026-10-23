
import json
import re
from typing import Any


class InvalidLLMResponseError(ValueError):
    """Raised when the model's output is malformed or violates the contract.

    Subclasses ValueError so existing callers keep working, but the API layer
    maps it to a provider-side failure (502) instead of a client error (422).
    """


ABSTENTION_MESSAGE = (
    "I could not find sufficient evidence in the retrieved passages "
    "to answer this question."
)


def build_answer_prompt(
    question: str,
    passages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Build a prompt with stable citation IDs and source metadata."""
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

Return exactly one JSON object, without Markdown fences or surrounding text.
Use exactly these fields:
- "answer": string
- "abstained": boolean
- "citations": array of citation ID strings

For an answer supported by the evidence, abstained must be false and
citations must list the source IDs used in the answer.
For an unsupported question, abstained must be true and citations must be [].

Question:
{question}

Evidence passages:
{evidence}
""".strip()

    return prompt, sources


def _parse_llm_json(raw_response: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating surrounding prose or Markdown fences."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise InvalidLLMResponseError("LLM returned an empty response")

    text = raw_response.strip()

    # First try the complete response.
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = None

    # Handle Markdown fences or short surrounding explanations.
    if result is None:
        unfenced = re.sub(
            r"\A```(?:json)?\s*|\s*```\Z",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        try:
            result = json.loads(unfenced)
        except json.JSONDecodeError:
            # Extract a JSON object only if the entire extracted object parses.
            decoder = json.JSONDecoder()
            result = None

            for match in re.finditer(r"\{", unfenced):
                try:
                    candidate, _ = decoder.raw_decode(unfenced[match.start():])
                except json.JSONDecodeError:
                    continue

                if isinstance(candidate, dict):
                    result = candidate
                    break

    if not isinstance(result, dict):
        raise InvalidLLMResponseError("LLM returned invalid JSON")

    return result


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
    result = _parse_llm_json(raw_response)

    answer = result.get("answer")
    abstained = result.get("abstained")
    citations = result.get("citations")

    if not isinstance(answer, str) or not answer.strip():
        raise InvalidLLMResponseError("LLM answer must be a non-empty string")
    if not isinstance(abstained, bool):
        raise InvalidLLMResponseError("LLM abstained field must be a boolean")
    if not isinstance(citations, list) or any(
        not isinstance(citation, str) for citation in citations
    ):
        raise InvalidLLMResponseError("LLM citations must be a list of strings")

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
        raise InvalidLLMResponseError("LLM returned unknown or duplicate citations")

    if not citations:
        raise InvalidLLMResponseError("A non-abstaining answer must cite at least one source")

    inline_ids = set(re.findall(r"\[(S\d+)\]", answer))
    if not set(citations).issubset(inline_ids):
        raise InvalidLLMResponseError("A non-abstaining answer must cite its sources inline")

    source_by_id = {source["citation_id"]: source for source in sources}

    return {
        "question": question,
        "answer": answer.strip(),
        "abstained": False,
        "citations": citations,
        "sources": [source_by_id[citation] for citation in citations],
    }
