"""
Hierarchical Chunker for long reports and documents.
Creates parent-child chunk relationships for better context.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class HierarchicalChunk:
    """Represents a chunk in a hierarchical structure."""
    content: str
    chunk_id: str
    level: str  # "parent" or "child"
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalChunker:
    """
    Hierarchical chunker for long documents.
    Creates small child chunks for retrieval and larger parent chunks for context.
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
        parent_overlap: int = 100,
        child_overlap: int = 50
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            parent_chunk_size: Size of parent chunks
            child_chunk_size: Size of child chunks
            parent_overlap: Overlap for parent chunks
            child_overlap: Overlap for child chunks
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(
        self,
        text: str,
        source: str = "unknown",
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, List[HierarchicalChunk]]:
        """
        Create hierarchical chunks from text.
        
        Args:
            text: Text to chunk
            source: Source identifier
            base_metadata: Additional metadata
            
        Returns:
            Dict with 'parents' and 'children' lists
        """
        base_metadata = base_metadata or {}
        
        # Create parent chunks
        parent_docs = self.parent_splitter.create_documents([text])
        
        parents = []
        children = []
        
        for p_idx, parent_doc in enumerate(parent_docs):
            parent_id = f"{source}_parent_{p_idx}"
            
            # Create parent chunk
            parent_metadata = {
                **base_metadata,
                "source": source,
                "chunk_type": "hierarchical_parent",
                "chunk_index": p_idx,
                "level": "parent"
            }
            
            parent = HierarchicalChunk(
                content=parent_doc.page_content,
                chunk_id=parent_id,
                level="parent",
                metadata=parent_metadata
            )
            
            # Create child chunks from parent content
            child_docs = self.child_splitter.create_documents([parent_doc.page_content])
            child_ids = []
            
            for c_idx, child_doc in enumerate(child_docs):
                child_id = f"{source}_child_{p_idx}_{c_idx}"
                child_ids.append(child_id)
                
                child_metadata = {
                    **base_metadata,
                    "source": source,
                    "chunk_type": "hierarchical_child",
                    "parent_id": parent_id,
                    "parent_index": p_idx,
                    "child_index": c_idx,
                    "level": "child"
                }
                
                children.append(HierarchicalChunk(
                    content=child_doc.page_content,
                    chunk_id=child_id,
                    level="child",
                    parent_id=parent_id,
                    metadata=child_metadata
                ))
            
            # Update parent with child IDs
            parent.child_ids = child_ids
            parent.metadata["child_count"] = len(child_ids)
            parents.append(parent)
        
        return {
            "parents": parents,
            "children": children
        }
    
    def get_parent_for_child(
        self, 
        child_id: str, 
        chunks: Dict[str, List[HierarchicalChunk]]
    ) -> Optional[HierarchicalChunk]:
        """
        Find the parent chunk for a given child chunk ID.
        
        Args:
            child_id: ID of the child chunk
            chunks: Hierarchical chunks dict
            
        Returns:
            Parent chunk or None
        """
        for child in chunks.get("children", []):
            if child.chunk_id == child_id and child.parent_id:
                for parent in chunks.get("parents", []):
                    if parent.chunk_id == child.parent_id:
                        return parent
        return None
    
    def format_with_parent_context(
        self,
        child: HierarchicalChunk,
        parent: HierarchicalChunk
    ) -> str:
        """
        Format child chunk with parent context for LLM.
        
        Args:
            child: Child chunk
            parent: Parent chunk
            
        Returns:
            Formatted context string
        """
        return f"""[Parent Context]
{parent.content}

[Specific Section]
{child.content}"""
