"""
Admin endpoints for system management.
"""

from fastapi import APIRouter, HTTPException, Depends

from app.config import get_settings, Settings
from app.models import StatsResponse, ClearResponse
from src.vectorstore import ChromaManager, EmbeddingsManager


router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(settings: Settings = Depends(get_settings)):
    """
    Get system statistics.
    
    - Total documents and chunks
    - Collections available
    - Chunks per collection
    """
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        stats = chroma.get_all_stats()
        
        return StatsResponse(
            total_documents=stats["total_collections"],
            total_chunks=stats["total_chunks"],
            collections=stats["collections"],
            chunks_per_collection=stats["chunks_per_collection"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", response_model=ClearResponse)
async def clear_database(
    confirm: bool = False,
    settings: Settings = Depends(get_settings)
):
    """
    Clear all data from ChromaDB.
    
    - Requires confirm=true parameter
    - Deletes all collections
    - Use with caution!
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm deletion of all data"
        )
    
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        cleared = chroma.clear_all()
        
        return ClearResponse(
            collections_cleared=cleared,
            message=f"Cleared {len(cleared)} collections"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    confirm: bool = False,
    settings: Settings = Depends(get_settings)
):
    """
    Delete a specific collection.
    
    - collection_name: Name of collection to delete
    - confirm: Must be true to confirm deletion
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail=f"Add ?confirm=true to confirm deletion of collection '{collection_name}'"
        )
    
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        # Check if collection exists
        collections = chroma.list_collections()
        if collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found. Available: {collections}"
            )
        
        success = chroma.delete_collection(collection_name)
        
        if success:
            return {
                "message": f"Successfully deleted collection '{collection_name}'",
                "collection": collection_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete collection")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def list_collections(settings: Settings = Depends(get_settings)):
    """List all available collections."""
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        collections = chroma.list_collections()
        
        return {
            "collections": collections,
            "count": len(collections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}")
async def view_collection(
    collection_name: str,
    limit: int = 10,
    settings: Settings = Depends(get_settings)
):
    """
    View chunks in a specific collection.
    
    - collection_name: Name of the collection to view
    - limit: Maximum number of chunks to return (default 10)
    """
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        collection = chroma.get_or_create_collection(collection_name)
        
        # Get all items from collection
        results = collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results and results["documents"]:
            for i, (doc, meta, chunk_id) in enumerate(zip(
                results["documents"], 
                results["metadatas"],
                results["ids"]
            )):
                chunks.append({
                    "id": chunk_id,
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "metadata": meta
                })
        
        return {
            "collection": collection_name,
            "total_in_collection": collection.count(),
            "showing": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
