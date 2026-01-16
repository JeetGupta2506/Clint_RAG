"""
RAG Retriever for fetching relevant documents.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.vectorstore.chroma_manager import ChromaManager, SearchResult


@dataclass
class RetrievedDocument:
    """A retrieved document with context."""
    content: str
    source: str
    chunk_id: str
    score: float
    page: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RAGRetriever:
    """
    Retriever for RAG system.
    Searches across collections and returns relevant documents.
    """
    
    def __init__(
        self,
        chroma_manager: ChromaManager,
        default_top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize retriever.
        
        Args:
            chroma_manager: ChromaManager instance
            default_top_k: Default number of results
            similarity_threshold: Minimum similarity score
        """
        self.chroma_manager = chroma_manager
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        website_context: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            website_context: Optional website to prioritize
            top_k: Number of results to return
            filter_dict: Metadata filter
            
        Returns:
            List of RetrievedDocument objects
        """
        top_k = top_k or self.default_top_k
        
        # Determine which collections to search
        collections = self._get_target_collections(website_context)
        print(f"🔍 Searching collections: {collections}")
        
        # Search
        results = self.chroma_manager.search(
            query=query,
            collection_names=collections,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        print(f"📄 Found {len(results)} results")
        
        # Convert to RetrievedDocument (no threshold filtering for now)
        documents = []
        for result in results:
            print(f"  - Score: {result.score:.4f}, Source: {result.metadata.get('source', 'unknown')}")
            documents.append(RetrievedDocument(
                content=result.content,
                source=result.metadata.get("source", "unknown"),
                chunk_id=result.chunk_id,
                score=result.score,
                page=result.metadata.get("page"),
                metadata=result.metadata
            ))
        
        return documents
    
    def _get_target_collections(self, website_context: Optional[str]) -> List[str]:
        """
        Determine which collections to search.
        
        Args:
            website_context: Optional website filter
            
        Returns:
            List of collection names
        """
        all_collections = self.chroma_manager.list_collections()
        
        if not all_collections:
            return [ChromaManager.DARUKA_COLLECTION]
        
        if website_context:
            # Search Daruka + specific website collection
            target = [ChromaManager.DARUKA_COLLECTION]
            
            # Find matching website collection
            website_coll = website_context.lower().replace(" ", "_")
            for coll in all_collections:
                if website_coll in coll.lower():
                    target.append(coll)
                    break
            
            return target
        
        # Search all collections
        return all_collections
    
    def format_context(self, documents: List[RetrievedDocument]) -> str:
        """
        Format retrieved documents as context for LLM.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source_info = f"[Source: {doc.source}"
            if doc.page:
                source_info += f", Page {doc.page}"
            source_info += "]"
            
            context_parts.append(f"--- Document {i} {source_info} ---\n{doc.content}")
        
        return "\n\n".join(context_parts)
    
    def get_sources_for_response(
        self, 
        documents: List[RetrievedDocument]
    ) -> List[Dict[str, Any]]:
        """
        Format documents as sources for API response.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of source dictionaries
        """
        return [
            {
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "source": doc.source,
                "page": doc.page,
                "chunk_id": doc.chunk_id,
                "score": round(doc.score, 4)
            }
            for doc in documents
        ]
