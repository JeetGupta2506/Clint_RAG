"""
Table Chunker for handling structured table data.
Stores entire tables as single chunks with context.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.processors.table_extractor import ExtractedTable, TableExtractor


@dataclass
class TableChunk:
    """Represents a table stored as a single chunk."""
    content: str
    chunk_id: str
    table: ExtractedTable
    metadata: Dict[str, Any] = field(default_factory=dict)


class TableChunker:
    """
    Chunker for table data.
    Stores entire tables as single chunks to preserve structure.
    """
    
    def __init__(self, max_table_size: int = 5000):
        """
        Initialize table chunker.
        
        Args:
            max_table_size: Maximum size before splitting large tables
        """
        self.max_table_size = max_table_size
        self.table_extractor = TableExtractor()
    
    def chunk(
        self, 
        table: ExtractedTable,
        source: str = "unknown",
        context: str = "",
        base_metadata: Dict[str, Any] = None
    ) -> List[TableChunk]:
        """
        Create chunk(s) from a table.
        
        Args:
            table: ExtractedTable to chunk
            source: Source identifier
            context: Surrounding text context
            base_metadata: Additional metadata
            
        Returns:
            List of TableChunk objects (usually 1 unless table is very large)
        """
        base_metadata = base_metadata or {}
        
        # Add context to table
        table.context = context
        
        # Format table for retrieval
        content = self.table_extractor.format_as_context(table)
        
        # Check if table needs to be split
        if len(content) > self.max_table_size:
            return self._split_large_table(table, source, base_metadata)
        
        chunk_id = f"{source}_table_{hash(content) % 10000}"
        
        metadata = {
            **base_metadata,
            "source": source,
            "chunk_type": "table",
            "rows": table.rows,
            "columns": table.columns,
            "table_title": table.title,
            "headers": table.metadata.get("headers", [])
        }
        
        return [TableChunk(
            content=content,
            chunk_id=chunk_id,
            table=table,
            metadata=metadata
        )]
    
    def chunk_from_data(
        self,
        data: List[List[Any]],
        headers: List[str] = None,
        title: str = None,
        source: str = "unknown",
        context: str = "",
        base_metadata: Dict[str, Any] = None
    ) -> List[TableChunk]:
        """
        Create chunk from raw table data.
        
        Args:
            data: 2D list of table data
            headers: Column headers
            title: Table title
            source: Source identifier
            context: Surrounding text
            base_metadata: Additional metadata
            
        Returns:
            List of TableChunk objects
        """
        table = self.table_extractor.convert_to_markdown(
            data=data,
            headers=headers,
            title=title
        )
        
        return self.chunk(
            table=table,
            source=source,
            context=context,
            base_metadata=base_metadata
        )
    
    def _split_large_table(
        self, 
        table: ExtractedTable, 
        source: str,
        base_metadata: Dict[str, Any]
    ) -> List[TableChunk]:
        """Split a large table into multiple chunks."""
        chunks = []
        
        # Parse the markdown table
        lines = table.markdown.split("\n")
        if len(lines) < 3:
            return []
        
        header = lines[0]
        separator = lines[1]
        rows = lines[2:]
        
        # Split rows into groups
        rows_per_chunk = max(10, len(rows) // 3)
        
        for i in range(0, len(rows), rows_per_chunk):
            chunk_rows = rows[i:i + rows_per_chunk]
            chunk_markdown = "\n".join([header, separator] + chunk_rows)
            
            part_num = i // rows_per_chunk + 1
            total_parts = (len(rows) + rows_per_chunk - 1) // rows_per_chunk
            
            chunk_table = ExtractedTable(
                markdown=chunk_markdown,
                description=f"{table.description} (Part {part_num}/{total_parts})",
                title=f"{table.title} (Part {part_num})" if table.title else None,
                rows=len(chunk_rows),
                columns=table.columns,
                context=table.context
            )
            
            chunk_id = f"{source}_table_{hash(chunk_markdown) % 10000}_part{part_num}"
            
            metadata = {
                **base_metadata,
                "source": source,
                "chunk_type": "table",
                "part": part_num,
                "total_parts": total_parts,
                "rows": len(chunk_rows),
                "columns": table.columns
            }
            
            chunks.append(TableChunk(
                content=self.table_extractor.format_as_context(chunk_table),
                chunk_id=chunk_id,
                table=chunk_table,
                metadata=metadata
            ))
        
        return chunks
