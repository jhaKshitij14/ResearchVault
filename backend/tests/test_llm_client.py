import httpx
import pytest

from app.services.llm_client import (
    LLMProviderError,
    OpenAICompatibleClient,
)


@pytest.mark.anyio
async def test_generate_sends_chat_completion_request():
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer": "Hello"}'}}
                ]
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        http_client=http,
    )

    try:
        result = await client.generate("Return JSON")
        assert result == '{"answer": "Hello"}'
    finally:
        await client.close()


@pytest.mark.anyio
async def test_provider_http_error_is_wrapped():
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, json={"error": "rate limited"})
        )
    )
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        http_client=http,
    )

    try:
        with pytest.raises(LLMProviderError, match="request failed"):
            await client.generate("Question")
    finally:
        await client.close()


@pytest.mark.anyio
async def test_rejects_empty_prompt():
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        http_client=http,
    )

    try:
        with pytest.raises(ValueError, match="Prompt must not be empty"):
            await client.generate("  ")
    finally:
        await client.close()


def test_rejects_empty_api_key():
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleClient(
            api_key=" ",
            base_url="https://provider.example/v1",
            model="test-model",
        )
