"""
Pydantic models for API request/response validation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """A source document with metadata."""
    content: str = Field(..., description="Content of the source chunk")
    source: str = Field(..., description="Source file or sheet name")
    page: Optional[int] = Field(None, description="Page number if from PDF")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Similarity score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    query: str = Field(..., description="User's question", min_length=1)
    website_context: Optional[str] = Field(
        None, 
        description="Optional website filter (e.g., 'website_a')"
    )
    top_k: int = Field(
        default=5, 
        description="Number of results to retrieve",
        ge=1, 
        le=20
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(
        default_factory=list, 
        description="Source documents used"
    )
    query: str = Field(..., description="Original query")


class UploadResponse(BaseModel):
    """Response model for /upload endpoint."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    chunks_created: int = Field(..., description="Number of chunks created")
    collection: str = Field(..., description="Collection stored in")
    message: str = Field(default="Upload successful", description="Status message")


class IngestResponse(BaseModel):
    """Response model for /ingest/sheets endpoint."""
    sheets_processed: int = Field(..., description="Number of sheets processed")
    total_chunks: int = Field(..., description="Total chunks created")
    collections: List[str] = Field(..., description="Collections updated")
    message: str = Field(default="Ingest successful", description="Status message")


class StatsResponse(BaseModel):
    """Response model for /stats endpoint."""
    total_documents: int = Field(..., description="Total documents ingested")
    total_chunks: int = Field(..., description="Total chunks stored")
    collections: List[str] = Field(..., description="Available collections")
    chunks_per_collection: dict = Field(
        default_factory=dict, 
        description="Chunk count per collection"
    )


class ClearResponse(BaseModel):
    """Response model for /clear endpoint."""
    collections_cleared: List[str] = Field(..., description="Collections cleared")
    message: str = Field(default="Clear successful", description="Status message")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="API version")
