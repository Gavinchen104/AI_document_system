"""Application configuration from environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    app_name: str = "AI Document System"
    debug: bool = False

    # LLM (OpenAI-compatible)
    openai_api_key: str = ""
    openai_base_url: str | None = None  # for local/open-source endpoints
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Storage
    data_dir: str = "./data"
    chroma_persist_dir: str = "./data/chroma"
    sqlite_path: str = "./data/store.db"

    # Optional: use local embeddings (sentence-transformers) instead of OpenAI
    use_local_embeddings: bool = False
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # Hybrid retrieval: alpha * BM25 + (1-alpha) * dense. Tuned on eval.
    hybrid_alpha: float = 0.3

    # Input limits
    max_file_size_mb: int = 10
    max_text_length: int = 50_000


settings = Settings()
