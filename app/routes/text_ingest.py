"""
Text ingestion endpoint for adding website content directly.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.config import get_settings, Settings
from src.chunking import ChunkingRouter
from src.vectorstore import ChromaManager, EmbeddingsManager


router = APIRouter()


class TextContent(BaseModel):
    """Single piece of text content to ingest."""
    content: str = Field(..., description="Text content to add")
    title: Optional[str] = Field(None, description="Title or label for this content")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")


class TextIngestRequest(BaseModel):
    """Request to ingest text content into a collection."""
    collection_name: str = Field(..., description="Name of the collection (e.g., 'reptors_org')")
    contents: List[TextContent] = Field(..., description="List of text contents to add")
    chunk_content: bool = Field(default=True, description="Whether to chunk the content")


class TextIngestResponse(BaseModel):
    """Response from text ingestion."""
    collection: str
    chunks_added: int
    message: str


@router.post("/ingest/text", response_model=TextIngestResponse)
async def ingest_text(
    request: TextIngestRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Ingest text content directly into a collection.
    
    Use this to add website content, FAQ data, or any text to the knowledge base.
    """
    import uuid
    
    try:
        # Initialize components
        embeddings = EmbeddingsManager(model=settings.embedding_model)
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        chunking_router = ChunkingRouter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # Sanitize collection name
        collection_name = request.collection_name.lower().replace(" ", "_").replace(".", "_")
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        doc_id = str(uuid.uuid4())[:8]
        chunk_counter = 0
        
        for i, text_item in enumerate(request.contents):
            if not text_item.content.strip():
                continue
            
            base_metadata = {
                "source": collection_name,
                "title": text_item.title or f"Content {i+1}",
                "type": "website",
                **(text_item.metadata or {})
            }
            
            if request.chunk_content:
                # Chunk the content
                result = chunking_router.route_and_chunk(
                    content=text_item.content,
                    source=collection_name,
                    base_metadata=base_metadata
                )
                
                for chunk in result.chunks:
                    all_documents.append(chunk.content)
                    all_metadatas.append(chunk.metadata)
                    all_ids.append(f"{doc_id}_chunk_{chunk_counter}")
                    chunk_counter += 1
            else:
                # Add as single document
                all_documents.append(text_item.content)
                all_metadatas.append(base_metadata)
                all_ids.append(f"{doc_id}_doc_{i}")
                chunk_counter += 1
        
        if not all_documents:
            raise HTTPException(status_code=400, detail="No valid content to ingest")
        
        # Store in ChromaDB
        count = chroma.add_documents(
            documents=all_documents,
            metadatas=all_metadatas,
            ids=all_ids,
            collection_name=collection_name
        )
        
        return TextIngestResponse(
            collection=collection_name,
            chunks_added=count,
            message=f"Successfully added {count} chunks to collection '{collection_name}'"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
