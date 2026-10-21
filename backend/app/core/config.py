from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+psycopg://researchvault:researchvault"
        "@localhost:5432/researchvault"
    )

    jina_api_key: str | None = None

    llm_api_key: str | None = None
    llm_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    llm_model: str = "gemini-2.5-flash"
    llm_timeout: float = 60.0

    upload_dir: Path = Path("storage/uploads")
    max_upload_bytes: int = 20 * 1024 * 1024

    model_config = SettingsConfigDict(
    env_file=".env",
    env_prefix="",
    case_sensitive=False,
)

settings = Settings()
