
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    app_name: str = Field(default="Smart Insurance Claim Advisor", validation_alias="APP_NAME")
    app_version: str = Field(default="1.0.0", validation_alias="APP_VERSION")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    astra_db_application_token: str = Field(validation_alias="ASTRA_DB_APPLICATION_TOKEN")
    astra_db_api_endpoint: str = Field(validation_alias="ASTRA_DB_API_ENDPOINT")
    astra_db_keyspace: str = Field(default="insurance_claims", validation_alias="ASTRA_DB_KEYSPACE")

    groq_api_key: str = Field(validation_alias="GROQ_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, validation_alias="HUGGINGFACE_API_KEY")

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL")
    chunk_size: int = Field(default=1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")

    llm_model: str = Field(default="gemma2-9b-it", validation_alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, validation_alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2048, validation_alias="MAX_TOKENS")

    vector_store_collection: str = Field(default="insurance_documents", validation_alias="VECTOR_STORE_COLLECTION")
    similarity_threshold: float = Field(default=0.6, validation_alias="SIMILARITY_THRESHOLD")
    max_search_results: int = Field(default=10, validation_alias="MAX_SEARCH_RESULTS")

    max_file_size: int = Field(default=50 * 1024 * 1024, validation_alias="MAX_FILE_SIZE")
    supported_file_types: List[str] = Field(default_factory=lambda: ["pdf", "docx", "pptx", "csv", "txt", "png", "jpg", "jpeg"], validation_alias="SUPPORTED_FILE_TYPES")

    tesseract_path: Optional[str] = Field(default=None, validation_alias="TESSERACT_PATH")
    conversation_memory_size: int = Field(default=50, validation_alias="CONVERSATION_MEMORY_SIZE")
    conversation_ttl: int = Field(default=86400, validation_alias="CONVERSATION_TTL")

    batch_size: int = Field(default=10, validation_alias="BATCH_SIZE")
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    retry_delay: int = Field(default=1, validation_alias="RETRY_DELAY")

    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", validation_alias="LOG_FORMAT")
    log_file: str = Field(default="insurance_advisor.log", validation_alias="LOG_FILE")

    debug_mode: bool = Field(default=False, validation_alias="DEBUG_MODE")
    verbose_logging: bool = Field(default=False, validation_alias="VERBOSE_LOGGING")

    rate_limit_requests: int = Field(default=100, validation_alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, validation_alias="RATE_LIMIT_WINDOW")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def validate_configuration(self) -> bool:
        if not self.astra_db_application_token:
            raise ValueError("ASTRA_DB_APPLICATION_TOKEN is required")
        if not self.astra_db_api_endpoint:
            raise ValueError("ASTRA_DB_API_ENDPOINT is required")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be positive")
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        return True

    def get_display_config(self) -> dict:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "log_level": self.log_level,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "max_tokens": self.max_tokens,
            "similarity_threshold": self.similarity_threshold,
            "max_search_results": self.max_search_results,
            "supported_file_types": self.supported_file_types,
            "conversation_memory_size": self.conversation_memory_size,
            "debug_mode": self.debug_mode
        }


# Create settings instance using environment variables and defaults
settings = Settings(
    astra_db_application_token=os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""),
    astra_db_api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT", ""),
    astra_db_keyspace=os.getenv("ASTRA_DB_KEYSPACE", "insurance_claims"),
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
    llm_model=os.getenv("LLM_MODEL", "gemma2-9b-it"),
    llm_temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
    max_tokens=int(os.getenv("MAX_TOKENS", 2048)),
    vector_store_collection=os.getenv("VECTOR_STORE_COLLECTION", "insurance_documents"),
    similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", 0.6)),
    max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", 10)),
    max_file_size=int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024)),
    supported_file_types=[ft.strip() for ft in os.getenv("SUPPORTED_FILE_TYPES", "pdf,docx,pptx,csv,txt,png,jpg,jpeg").split(",")],
    tesseract_path=os.getenv("TESSERACT_PATH"),
    conversation_memory_size=int(os.getenv("CONVERSATION_MEMORY_SIZE", 50)),
    conversation_ttl=int(os.getenv("CONVERSATION_TTL", 86400)),
    batch_size=int(os.getenv("BATCH_SIZE", 10)),
    max_retries=int(os.getenv("MAX_RETRIES", 3)),
    retry_delay=int(os.getenv("RETRY_DELAY", 1)),
    log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    log_file=os.getenv("LOG_FILE", "insurance_advisor.log"),
    debug_mode=os.getenv("DEBUG_MODE", "False").lower() == "true",
    verbose_logging=os.getenv("VERBOSE_LOGGING", "False").lower() == "true",
    rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", 100)),
    rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", 3600)),
    app_name=os.getenv("APP_NAME", "Smart Insurance Claim Advisor"),
    app_version=os.getenv("APP_VERSION", "1.0.0"),
    log_level=os.getenv("LOG_LEVEL", "INFO")
)
try:
    settings.validate_configuration()
except Exception as e:
    print(f"Configuration Error: {e}")
    exit(1)