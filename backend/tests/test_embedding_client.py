
import asyncio

import httpx
import pytest

from app.services.embedding_client import (
    EmbeddingError,
    EXPECTED_DIMENSIONS,
    JinaEmbeddingClient,
)


def make_client(handler):
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)

    client = JinaEmbeddingClient(
        api_key="test-key",
        http_client=http_client,
    )
    return client, http_client


def test_embeds_text_and_sends_passage_task():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-key"

        import json
        payload = json.loads(request.content)
        assert payload["model"] == "jina-embeddings-v3"
        assert payload["task"] == "retrieval.passage"
        assert payload["input"] == ["A database index"]

        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "embedding": [0.1] * EXPECTED_DIMENSIONS,
                    }
                ]
            },
        )

    client, http_client = make_client(handler)

    async def run():
        try:
            vectors = await client.embed(["A database index"])
            assert len(vectors) == 1
            assert len(vectors[0]) == EXPECTED_DIMENSIONS
        finally:
            await http_client.aclose()

    asyncio.run(run())


def test_sorts_vectors_by_input_index():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2] * EXPECTED_DIMENSIONS},
                    {"index": 0, "embedding": [0.1] * EXPECTED_DIMENSIONS},
                ]
            },
        )

    client, http_client = make_client(handler)

    async def run():
        try:
            vectors = await client.embed(["first", "second"])
            assert vectors[0][0] == 0.1
            assert vectors[1][0] == 0.2
        finally:
            await http_client.aclose()

    asyncio.run(run())


def test_rejects_wrong_vector_dimensions():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]}
                ]
            },
        )

    client, http_client = make_client(handler)

    async def run():
        try:
            with pytest.raises(EmbeddingError, match="dimensional"):
                await client.embed(["test"])
        finally:
            await http_client.aclose()

    asyncio.run(run())


def test_wraps_http_errors():
    def handler(request):
        return httpx.Response(429, json={"detail": "Rate limit exceeded"})

    client, http_client = make_client(handler)

    async def run():
        try:
            with pytest.raises(EmbeddingError, match="Failed to generate"):
                await client.embed(["test"])
        finally:
            await http_client.aclose()

    asyncio.run(run())


def test_rejects_empty_text():
    client, http_client = make_client(
        lambda request: httpx.Response(200, json={"data": []})
    )

    async def run():
        try:
            with pytest.raises(ValueError, match="empty strings"):
                await client.embed(["  "])
        finally:
            await http_client.aclose()

    asyncio.run(run())
