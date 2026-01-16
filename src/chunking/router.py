"""
Chunking Router for automatically selecting the appropriate chunking strategy.
Routes content based on detected type (table, Q&A, long report, or narrative).
"""

from typing import List, Dict, Any, Union
from dataclasses import dataclass
import re

from .semantic_chunker import SemanticChunker, Chunk
from .table_chunker import TableChunker, TableChunk
from .qa_chunker import QAChunker, QAChunk
from .hierarchical_chunker import HierarchicalChunker, HierarchicalChunk
from src.processors.table_extractor import ExtractedTable


@dataclass
class ChunkResult:
    """Result from chunking router."""
    chunks: List[Union[Chunk, TableChunk, QAChunk, HierarchicalChunk]]
    strategy_used: str
    parent_chunks: List[HierarchicalChunk] = None  # For hierarchical chunking


class ChunkingRouter:
    """
    Automatically routes content to the appropriate chunking strategy.
    
    Priority order:
    1. Tables → Table Chunker
    2. Q&A pairs → Q&A Chunker
    3. Long reports → Hierarchical Chunker
    4. Everything else → Semantic Chunker
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        hierarchical_threshold: int = 5000
    ):
        """
        Initialize chunking router with all chunkers.
        
        Args:
            chunk_size: Default chunk size for semantic chunking
            chunk_overlap: Overlap between chunks
            hierarchical_threshold: Text length threshold for hierarchical chunking
        """
        self.semantic_chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.table_chunker = TableChunker()
        self.qa_chunker = QAChunker()
        self.hierarchical_chunker = HierarchicalChunker()
        
        self.hierarchical_threshold = hierarchical_threshold
    
    def route_and_chunk(
        self,
        content: Any,
        content_type: str = "auto",
        source: str = "unknown",
        website: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> ChunkResult:
        """
        Automatically detect content type and apply appropriate chunking.
        
        Args:
            content: Content to chunk (str, ExtractedTable, or dict)
            content_type: Explicit type ("auto", "text", "table", "qa", "hierarchical")
            source: Source identifier
            website: Website source name
            base_metadata: Additional metadata
            
        Returns:
            ChunkResult with chunks and strategy used
        """
        base_metadata = base_metadata or {}
        
        # Handle explicit table type
        if content_type == "table" or isinstance(content, ExtractedTable):
            return self._chunk_table(content, source, base_metadata)
        
        # Handle explicit Q&A type
        if content_type == "qa":
            return self._chunk_qa(content, source, website, base_metadata)
        
        # Handle hierarchical type
        if content_type == "hierarchical":
            return self._chunk_hierarchical(content, source, base_metadata)
        
        # Auto-detect for text content
        if isinstance(content, str):
            return self._auto_detect_and_chunk(content, source, website, base_metadata)
        
        # Handle dict content (e.g., from sheets)
        if isinstance(content, dict):
            return self._handle_dict_content(content, source, website, base_metadata)
        
        # Handle list content (e.g., rows)
        if isinstance(content, list):
            return self._handle_list_content(content, source, website, base_metadata)
        
        # Fallback to semantic
        return self._chunk_semantic(str(content), source, base_metadata)
    
    def _auto_detect_and_chunk(
        self,
        text: str,
        source: str,
        website: str,
        base_metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Auto-detect content type and chunk accordingly."""
        
        # Check for table patterns
        if self._is_table(text):
            return ChunkResult(
                chunks=[],  # Tables should be passed as ExtractedTable
                strategy_used="table_detected"
            )
        
        # Check for Q&A patterns
        if self._is_qa_format(text):
            chunks = self.qa_chunker.chunk_from_text(
                text=text,
                source=source,
                website=website,
                base_metadata=base_metadata
            )
            if chunks:
                return ChunkResult(chunks=chunks, strategy_used="qa")
        
        # Check for long document
        if len(text) > self.hierarchical_threshold:
            return self._chunk_hierarchical(text, source, base_metadata)
        
        # Default to semantic
        return self._chunk_semantic(text, source, base_metadata)
    
    def _chunk_semantic(
        self, 
        text: str, 
        source: str, 
        base_metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Apply semantic chunking."""
        chunks = self.semantic_chunker.chunk(
            text=text,
            source=source,
            base_metadata=base_metadata
        )
        return ChunkResult(chunks=chunks, strategy_used="semantic")
    
    def _chunk_table(
        self, 
        content: Any, 
        source: str, 
        base_metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Apply table chunking."""
        if isinstance(content, ExtractedTable):
            chunks = self.table_chunker.chunk(
                table=content,
                source=source,
                base_metadata=base_metadata
            )
        else:
            # Assume it's raw data
            chunks = self.table_chunker.chunk_from_data(
                data=content,
                source=source,
                base_metadata=base_metadata
            )
        return ChunkResult(chunks=chunks, strategy_used="table")
    
    def _chunk_qa(
        self,
        content: Any,
        source: str,
        website: str,
        base_metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Apply Q&A chunking."""
        if isinstance(content, str):
            chunks = self.qa_chunker.chunk_from_text(
                text=content,
                source=source,
                website=website,
                base_metadata=base_metadata
            )
        elif isinstance(content, list) and content:
            if isinstance(content[0], dict):
                chunks = self.qa_chunker.chunk_from_rows(
                    rows=content,
                    source=source,
                    website=website,
                    base_metadata=base_metadata
                )
            else:
                chunks = self.qa_chunker.chunk(
                    qa_pairs=content,
                    source=source,
                    website=website,
                    base_metadata=base_metadata
                )
        else:
            chunks = []
        
        return ChunkResult(chunks=chunks, strategy_used="qa")
    
    def _chunk_hierarchical(
        self, 
        text: str, 
        source: str, 
        base_metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Apply hierarchical chunking."""
        result = self.hierarchical_chunker.chunk(
            text=text,
            source=source,
            base_metadata=base_metadata
        )
        return ChunkResult(
            chunks=result["children"],
            strategy_used="hierarchical",
            parent_chunks=result["parents"]
        )
    
    def _handle_dict_content(
        self,
        content: dict,
        source: str,
        website: str,
        base_metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Handle dictionary content."""
        # Check if it looks like a single row
        if all(isinstance(v, (str, int, float)) for v in content.values()):
            # Convert to text
            text = "\n".join(f"{k}: {v}" for k, v in content.items())
            return self._chunk_semantic(text, source, base_metadata)
        
        return ChunkResult(chunks=[], strategy_used="unknown")
    
    def _handle_list_content(
        self,
        content: list,
        source: str,
        website: str,
        base_metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Handle list content (rows from sheets)."""
        if not content:
            return ChunkResult(chunks=[], strategy_used="empty")
        
        # Check if it looks like rows with Q&A format
        if isinstance(content[0], dict):
            keys = list(content[0].keys())
            keys_lower = [k.lower() for k in keys]
            
            # Check for Q&A columns
            has_q = any("question" in k or k == "q" for k in keys_lower)
            has_a = any("answer" in k or k == "a" for k in keys_lower)
            
            if has_q and has_a:
                return self._chunk_qa(content, source, website, base_metadata)
        
        # Default to semantic for each row
        all_chunks = []
        for i, row in enumerate(content):
            if isinstance(row, dict):
                text = "\n".join(f"{k}: {v}" for k, v in row.items())
            else:
                text = str(row)
            
            chunks = self.semantic_chunker.chunk(
                text=text,
                source=f"{source}_row_{i}",
                base_metadata=base_metadata
            )
            all_chunks.extend(chunks)
        
        return ChunkResult(chunks=all_chunks, strategy_used="semantic")
    
    def _is_table(self, text: str) -> bool:
        """Detect if text contains table patterns."""
        table_patterns = [
            r'\|.*\|.*\|',  # Markdown table
            r'(?:\S+\s*\t){2,}\S+',  # Tab-separated with 3+ columns
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _is_qa_format(self, text: str) -> bool:
        """Detect if text is in Q&A format."""
        qa_patterns = [
            r'Q:\s*.+\s*A:',
            r'Question:\s*.+\s*Answer:',
            r'\*\*Question\*\*:',
        ]
        
        for pattern in qa_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
