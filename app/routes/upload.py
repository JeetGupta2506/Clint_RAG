"""
Upload endpoint for document ingestion.
"""

import os
import uuid
from typing import Union
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from app.config import get_settings, Settings
from app.models import UploadResponse
from src.processors import PDFProcessor
from src.chunking import ChunkingRouter
from src.vectorstore import ChromaManager, EmbeddingsManager


router = APIRouter()


def get_chroma_manager(settings: Settings = Depends(get_settings)) -> ChromaManager:
    """Dependency to get ChromaManager instance."""
    embeddings = EmbeddingsManager(
        model=settings.embedding_model
    )
    return ChromaManager(
        persist_directory=settings.chroma_db_path,
        embeddings_manager=embeddings
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = None,
    settings: Settings = Depends(get_settings)
):
    """
    Upload and process a document.
    
    - Supports PDF files
    - Automatically chunks content based on type
    - Stores in ChromaDB
    """
    # Validate file type
    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()
    
    if extension not in [".pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported: .pdf"
        )
    
    # Save file temporarily
    document_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.upload_dir, f"{document_id}{extension}")
    
    try:
        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        # Save uploaded file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process based on type
        if extension == ".pdf":
            chunks = await process_pdf(
                temp_path, 
                filename, 
                settings,
                collection
            )
        else:
            chunks = []
        
        # Clean up temp file
        os.remove(temp_path)
        
        target_collection = collection or ChromaManager.DARUKA_COLLECTION
        
        return UploadResponse(
            document_id=document_id,
            filename=filename,
            chunks_created=chunks,
            collection=target_collection,
            message=f"Successfully processed {filename}"
        )
        
    except Exception as e:
        # Clean up on error
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


async def process_pdf(
    file_path: str, 
    filename: str, 
    settings: Settings,
    collection: str = None,
    document_id: str = None
) -> int:
    """Process a PDF file and store chunks."""
    import uuid
    
    # Generate unique document ID
    doc_id = document_id or str(uuid.uuid4())[:8]
    
    # Extract text from PDF
    processor = PDFProcessor()
    pages = processor.process(file_path)
    
    # Initialize chunking router
    chunking_router = ChunkingRouter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Initialize vector store (free local embeddings)
    embeddings = EmbeddingsManager(
        model=settings.embedding_model
    )
    chroma = ChromaManager(
        persist_directory=settings.chroma_db_path,
        embeddings_manager=embeddings
    )
    
    all_chunks = []
    
    # Process each page
    for page in pages:
        if not page.content.strip():
            continue
        
        # Route and chunk content
        result = chunking_router.route_and_chunk(
            content=page.content,
            source=filename,
            base_metadata={
                "source": filename,
                "page": page.page_number,
                "type": "pdf",
                "document_id": doc_id
            }
        )
        
        all_chunks.extend(result.chunks)
    
    if not all_chunks:
        return 0
    
    # Prepare for storage with truly unique IDs
    documents = [chunk.content for chunk in all_chunks]
    metadatas = [chunk.metadata for chunk in all_chunks]
    # Use enumerate to ensure unique IDs across all chunks
    ids = [f"{doc_id}_chunk_{i}" for i, chunk in enumerate(all_chunks)]
    
    # Store in ChromaDB
    target_collection = collection or ChromaManager.DARUKA_COLLECTION
    count = chroma.add_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        collection_name=target_collection
    )
    
    return count
