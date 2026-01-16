"""
Configuration module for Daruka.Earth RAG System.
Loads and validates environment variables using Pydantic Settings.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Anthropic Configuration (for LLM)
    anthropic_api_key: str = Field(..., description="Anthropic API Key")
    
    # Google Sheets Configuration
    google_sheets_credentials_path: str = Field(
        default="./credentials/google_service_account.json",
        description="Path to Google service account credentials JSON"
    )
    google_sheet_id: str = Field(
        default="",
        description="Google Sheet ID to sync"
    )
    
    # ChromaDB Configuration
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        description="Path to ChromaDB persistent storage"
    )
    
    # Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model (runs locally, free)"
    )
    llm_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="LLM model name for generation"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(
        default=800,
        description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=150,
        description="Overlap between chunks in tokens"
    )
    
    # Retrieval Configuration
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for retrieval"
    )
    
    # Upload Configuration
    upload_dir: str = Field(
        default="./data/uploads",
        description="Directory for temporary file uploads"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
