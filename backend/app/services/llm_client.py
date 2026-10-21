
import asyncio
import random

import httpx


class LLMProviderError(RuntimeError):
    """Raised when the LLM provider request fails."""


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key,
        base_url,
        model,
        timeout=60.0,
        http_client=None,
    ):
        if not api_key.strip():
            raise ValueError("LLM API key must not be empty")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._owns_client = http_client is None
        self.client = http_client or httpx.AsyncClient(timeout=timeout)

    async def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        for attempt in range(3):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Follow the supplied instructions and "
                                    "return the requested output format."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                    },
                )
                response.raise_for_status()

                payload = response.json()
                content = payload["choices"][0]["message"]["content"]

                if not isinstance(content, str) or not content.strip():
                    raise LLMProviderError(
                        "LLM provider returned empty or invalid content"
                    )

                return content

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (
                    408, 429, 500, 502, 503, 504
                ):
                    raise LLMProviderError(
                        "LLM provider request failed"
                    ) from exc

                if attempt == 2:
                    raise LLMProviderError(
                        "LLM provider request failed after retries"
                    ) from exc

                retry_after = exc.response.headers.get("Retry-After")

                try:
                    delay = (
                        float(retry_after)
                        if retry_after
                        else 2**attempt
                    )
                except ValueError:
                    delay = 2**attempt

                await asyncio.sleep(
                    min(delay, 10) + random.uniform(0, 0.5)
                )

            except LLMProviderError:
                raise

            except (
                httpx.HTTPError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                raise LLMProviderError(
                    "LLM provider request failed or returned an invalid response"
                ) from exc

        raise LLMProviderError("LLM provider request failed")

    async def close(self):
        if self._owns_client:
            await self.client.aclose()
