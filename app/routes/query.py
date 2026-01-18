"""
Unified Query endpoint for RAG-based question answering with automatic project matching.
"""

from fastapi import APIRouter, HTTPException, Depends

from app.config import get_settings, Settings
from app.models import QueryRequest, QueryResponse, SourceDocument
from src.vectorstore import ChromaManager, EmbeddingsManager
from src.rag import RAGRetriever, RAGChain, get_memory
from src.rag.project_matcher import ProjectMatcher


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Unified query endpoint with automatic project matching.
    
    Features:
    - Searches all relevant collections
    - Automatically fetches matching Daruka projects when relevant
    - Maintains conversation memory per session_id + website_context
    - Returns answer with sources
    
    **For multi-turn conversations:**
    Use the same `session_id` across requests to maintain context.
    """
    try:
        # Initialize components
        embeddings = EmbeddingsManager(model=settings.embedding_model)
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        retriever = RAGRetriever(
            chroma_manager=chroma,
            default_top_k=request.top_k,
            similarity_threshold=settings.similarity_threshold
        )
        
        # Initialize project matcher
        matcher = ProjectMatcher(
            chroma_manager=chroma,
            api_key=settings.anthropic_api_key,
            model=settings.llm_model
        )
        
        rag_chain = RAGChain(
            retriever=retriever,
            api_key=settings.anthropic_api_key,
            model=settings.llm_model
        )
        
        # Get conversation history
        conversation_history = ""
        if request.session_id:
            memory = get_memory()
            conversation_history = memory.get_formatted_history(
                session_id=request.session_id,
                website_context=request.website_context or "default",
                max_messages=6
            )
        
        # Retrieve relevant documents
        documents = retriever.retrieve(
            query=request.query,
            website_context=request.website_context,
            top_k=request.top_k
        )
        rag_context = retriever.format_context(documents)
        
        # Automatically search for or generate matching projects
        project_context = ""
        project_info = None
        matching_project = None
        
        # Check if query is about projects/proposals/grants
        project_keywords = ["project", "proposal", "methodology", "approach", "plan", 
                           "describe", "objectives", "outcomes", "conservation", "monitoring"]
        query_lower = request.query.lower()
        is_project_query = any(kw in query_lower for kw in project_keywords)
        
        if is_project_query:
            # Try to find a matching project
            matching_project = matcher.find_matching_project(
                grant_focus=request.query,
                grant_requirements=rag_context,
                top_k=2
            )
            
            # If no match found, generate a hypothetical project
            if not matching_project:
                print(f"🔧 No matching project found. Generating hypothetical project...")
                # Determine grant focus from website context or query
                grant_focus = request.website_context.replace("_", " ").title() if request.website_context else "Conservation"
                
                matching_project = matcher.generate_hypothetical_project(
                    grant_focus=grant_focus,
                    grant_requirements=rag_context,
                    grant_context=request.website_context or ""
                )
                print(f"✨ Generated project: {matching_project.name}")
            else:
                print(f"✅ Found matching project: {matching_project.name} (score: {matching_project.relevance_score:.2f})")
        
        if matching_project:
            project_context = f"""
=== {'EXISTING' if matching_project.project_type == 'existing' else 'PROPOSED'} DARUKA PROJECT ===
Project: {matching_project.name}
Type: {matching_project.project_type.upper()}
Focus: {', '.join(matching_project.focus_areas)}
Location: {matching_project.location}
Description: {matching_project.description}
Methodology: {matching_project.methodology}
Expected Outcomes: {', '.join(matching_project.expected_outcomes)}
=== END PROJECT ===


"""
            project_info = {
                "name": matching_project.name,
                "type": matching_project.project_type,
                "focus_areas": matching_project.focus_areas,
                "relevance_score": matching_project.relevance_score
            }
        
        # Combine all context
        full_context = f"{project_context}{rag_context}"
        
        # Generate answer with custom context
        answer = rag_chain.query_with_custom_context(
            question=request.query,
            context=full_context,
            session_id=request.session_id,
            website_context=request.website_context
        )
        
        # Format sources
        sources = [
            SourceDocument(
                content=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                source=doc.source,
                page=doc.page,
                chunk_id=doc.chunk_id,
                score=doc.score,
                metadata={"project": project_info} if project_info else {}
            )
            for doc in documents
        ]
        
        # Add project as a source if found
        if project_info:
            sources.insert(0, SourceDocument(
                content=f"Project: {matching_project.name} - {matching_project.description[:300]}...",
                source="daruka_projects",
                page=None,
                chunk_id=matching_project.source_chunk_id or "project_match",
                score=matching_project.relevance_score,
                metadata=project_info
            ))
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            session_id=request.session_id
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
