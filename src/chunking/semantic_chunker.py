"""
Semantic Chunker using LangChain's RecursiveCharacterTextSplitter.
For narrative text with configurable chunk size and overlap.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    chunk_id: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: str = None


class SemanticChunker:
    """
    Semantic text chunker using recursive character splitting.
    Best for narrative text, articles, and general content.
    """
    
    def __init__(
        self, 
        chunk_size: int = 800, 
        chunk_overlap: int = 150,
        separators: List[str] = None
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def chunk(
        self, 
        text: str, 
        source: str = "unknown",
        base_metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to chunk
            source: Source identifier for the text
            base_metadata: Additional metadata to include in all chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        base_metadata = base_metadata or {}
        
        # Split text into documents
        docs = self.splitter.create_documents([text])
        
        chunks = []
        for i, doc in enumerate(docs):
            chunk_id = f"{source}_chunk_{i}"
            
            metadata = {
                **base_metadata,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(docs),
                "chunk_type": "semantic",
                "char_count": len(doc.page_content)
            }
            
            chunks.append(Chunk(
                content=doc.page_content,
                chunk_id=chunk_id,
                index=i,
                metadata=metadata
            ))
        
        return chunks
    
    def chunk_with_pages(
        self, 
        pages: List[Dict[str, Any]], 
        source: str = "unknown"
    ) -> List[Chunk]:
        """
        Chunk content from multiple pages while preserving page metadata.
        
        Args:
            pages: List of page dicts with 'content' and 'page' keys
            source: Source file identifier
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        global_index = 0
        
        for page_info in pages:
            content = page_info.get("content", "")
            page_num = page_info.get("page", 0)
            page_metadata = page_info.get("metadata", {})
            
            if not content.strip():
                continue
            
            page_chunks = self.chunk(
                content, 
                source=source,
                base_metadata={
                    **page_metadata,
                    "page": page_num
                }
            )
            
            # Update global indices
            for chunk in page_chunks:
                chunk.index = global_index
                chunk.chunk_id = f"{source}_chunk_{global_index}"
                chunk.metadata["global_index"] = global_index
                global_index += 1
            
            all_chunks.extend(page_chunks)
        
        # Update total chunks count
        for chunk in all_chunks:
            chunk.metadata["total_chunks"] = len(all_chunks)
        
        return all_chunks
