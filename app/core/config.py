from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Metadata
    API_TITLE: str = "CogniFin AI"
    API_DESCRIPTION: str = "Enterprise Financial RAG API for Indian Document Analysis"
    API_VERSION: str = "3.0.0"
    ENVIRONMENT: str = "production"
    ENABLE_DOCS: bool = False
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    # Backend Core & Retrieval
    PDF_PATH: str = "data/sample.pdf"
    INDEX_CACHE_DIR: str = "index_cache"
    ASSET_MODE: str = "local"
    HF_CACHE_URL: str | None = None
    HF_REPO_ID: str = "TheLunatic078/cognifin-assets"
    HF_TOKEN: str | None = None
    SPACE_ID: str | None = None

    @property
    def is_hf_space(self) -> bool:
        return bool(self.SPACE_ID)

    @property
    def asset_mode_resolved(self) -> str:
        if self.ASSET_MODE in ("local", "huggingface"):
            return self.ASSET_MODE
        return "huggingface" if self.is_hf_space else "local"

    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    RETRIEVAL_K: int = 10
    FINAL_K: int = 5
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30
    QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages:"

    # Cross-encoder Reranking
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_MAX_CANDIDATES: int = 5
    SKIP_RERANKER_THRESHOLD: float = 0.85

    # Refinement & Boosting
    BOOST_COMPANY: float = 0.08
    BOOST_YEAR: float = 0.04
    BOOST_DOCTYPE: float = 0.03
    DEDUP_THRESHOLD: float = 0.85
    MAX_FROM_ONE_DOC: float = 0.6
    CONTEXT_WINDOW: int = 0

    # Defaults
    DEFAULT_COMPANY: str = "demo_company"
    DEFAULT_DOC_TYPE: str = "DRHP"
    DEFAULT_YEAR: str = "2024"

    # Hybrid Search & Multi-query
    MULTI_QUERY_ENABLED: bool = False
    MULTI_QUERY_COUNT: int = 3
    BM25_ENABLED: bool = True
    BM25_WEIGHT: float = 0.3
    QUERY_EXPANSION_LLM: bool = False
    RRF_K: int = 60
    MAX_DOC_CONCENTRATION: float = 0.4

    # Intelligent Query Understanding
    INTELLIGENT_PARSING_ENABLED: bool = False
    MAX_RETRIEVAL_STEPS: int = 4
    MULTI_STEP_RETRIEVAL_K: int = 10

    # LLM & Generation (Google GenAI SDK)
    LLM_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.0-flash"
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Database
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "finsightai"

    # Authentication & Security
    JWT_SECRET: str = "change-me-to-a-strong-random-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/callback"
    FRONTEND_URL: str = "http://localhost:5173"

    # Optimization & Monitoring
    CACHE_ENABLED: bool = True
    CACHE_MAX_SIZE: int = 500
    CACHE_TTL_SECONDS: int = 3600
    QUERY_LOG_ENABLED: bool = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
