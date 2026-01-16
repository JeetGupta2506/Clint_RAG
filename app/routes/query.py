"""
Query endpoint for RAG-based question answering.
"""

from fastapi import APIRouter, HTTPException, Depends

from app.config import get_settings, Settings
from app.models import QueryRequest, QueryResponse, SourceDocument
from src.vectorstore import ChromaManager, EmbeddingsManager
from src.rag import RAGRetriever, RAGChain


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Query the knowledge base using RAG.
    
    - Searches relevant collections based on website_context
    - Uses LLM to generate contextual answers
    - Returns answer with source citations
    """
    try:
        # Initialize components (free local embeddings)
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        retriever = RAGRetriever(
            chroma_manager=chroma,
            default_top_k=request.top_k,
            similarity_threshold=settings.similarity_threshold
        )
        
        rag_chain = RAGChain(
            retriever=retriever,
            api_key=settings.anthropic_api_key,
            model=settings.llm_model
        )
        
        # Execute query
        result = rag_chain.query(
            question=request.query,
            website_context=request.website_context,
            top_k=request.top_k
        )
        
        # Format sources
        sources = [
            SourceDocument(
                content=src["content"],
                source=src["source"],
                page=src.get("page"),
                chunk_id=src["chunk_id"],
                score=src["score"],
                metadata={}
            )
            for src in result.sources
        ]
        
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
