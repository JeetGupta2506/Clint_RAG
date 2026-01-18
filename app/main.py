"""
FastAPI main application entry point for Daruka.Earth RAG System.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import upload, query, ingest, admin, text_ingest, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    settings = get_settings()
    
    # Create necessary directories on startup
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_db_path, exist_ok=True)
    os.makedirs(os.path.dirname(settings.google_sheets_credentials_path), exist_ok=True)
    
    print(f"📁 Upload directory: {settings.upload_dir}")
    print(f"📁 ChromaDB path: {settings.chroma_db_path}")
    print(f"🤖 Embedding model: {settings.embedding_model}")
    print(f"🧠 LLM model: {settings.llm_model}")
    print("✅ Daruka.Earth RAG System started successfully!")
    
    yield
    
    print("👋 Shutting down Daruka.Earth RAG System...")


app = FastAPI(
    title="Daruka.Earth RAG API",
    description="RAG (Retrieval-Augmented Generation) API for Daruka.Earth knowledge base",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(query.router, prefix="/api", tags=["Chat"])
app.include_router(ingest.router, prefix="/api", tags=["Ingest"])
app.include_router(text_ingest.router, prefix="/api", tags=["Text Ingest"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(admin.router, prefix="/api", tags=["Admin"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "service": "Daruka.Earth RAG API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
