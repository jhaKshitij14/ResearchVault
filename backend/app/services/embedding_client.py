
import httpx


JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_MODEL = "jina-embeddings-v3"
EXPECTED_DIMENSIONS = 1024


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class JinaEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        if not api_key.strip():
            raise ValueError("Jina API key must not be empty")

        self.model = model
        self.timeout = timeout
        self._owns_client = http_client is None

        self.client = http_client or httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

        # Ensure an injected client also sends the authorization header.
        self.client.headers["Authorization"] = f"Bearer {api_key}"

    async def embed(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Texts must not contain empty strings")

        if task not in {"retrieval.passage", "retrieval.query"}:
            raise ValueError("Unsupported embedding task")

        payload = {
            "model": self.model,
            "task": task,
            "input": texts,
        }

        try:
            response = await self.client.post(
                JINA_EMBEDDINGS_URL,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()

            data = body["data"]
            if len(data) != len(texts):
                raise EmbeddingError(
                    "Embedding response count does not match input count"
                )

            data = sorted(data, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]

            if any(
                len(vector) != EXPECTED_DIMENSIONS
                for vector in vectors
            ):
                raise EmbeddingError(
                    f"Expected {EXPECTED_DIMENSIONS}-dimensional vectors"
                )

            return vectors

        except EmbeddingError:
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(
                "Failed to generate embeddings with Jina"
            ) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
