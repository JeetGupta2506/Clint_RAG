"""
Table Extractor for detecting and converting tables to markdown format.
Handles tables from PDFs and spreadsheets.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExtractedTable:
    """Represents an extracted table."""
    markdown: str
    description: str
    title: Optional[str]
    rows: int
    columns: int
    context: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TableExtractor:
    """
    Extracts tables from text and converts them to markdown format.
    Generates natural language descriptions for tables.
    """
    
    def __init__(self):
        """Initialize table extractor."""
        self.table_patterns = [
            r'\|.*\|',  # Markdown-style tables
            r'(?:\S+\s*\t)+\S+',  # Tab-separated
        ]
    
    def extract_tables_from_text(self, text: str) -> List[ExtractedTable]:
        """
        Detect and extract tables from text content.
        
        Args:
            text: Text content potentially containing tables
            
        Returns:
            List of ExtractedTable objects
        """
        tables = []
        
        # Try to find markdown-style tables
        md_tables = self._find_markdown_tables(text)
        tables.extend(md_tables)
        
        # Try to find tab-separated tables
        tsv_tables = self._find_tsv_tables(text)
        tables.extend(tsv_tables)
        
        return tables
    
    def convert_to_markdown(
        self, 
        data: List[List[Any]], 
        headers: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> ExtractedTable:
        """
        Convert 2D data to markdown table format.
        
        Args:
            data: 2D list of table data
            headers: Optional column headers (uses first row if not provided)
            title: Optional table title
            
        Returns:
            ExtractedTable object
        """
        if not data:
            return ExtractedTable(
                markdown="",
                description="Empty table",
                title=title,
                rows=0,
                columns=0
            )
        
        # Use first row as headers if not provided
        if headers is None:
            headers = [str(h) for h in data[0]]
            data = data[1:]
        
        num_cols = len(headers)
        num_rows = len(data)
        
        # Build markdown table
        lines = []
        
        # Header row
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        lines.append(header_row)
        
        # Separator row
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        lines.append(separator)
        
        # Data rows
        for row in data:
            # Pad or truncate row to match headers
            cells = []
            for i in range(num_cols):
                cell = str(row[i]) if i < len(row) else ""
                # Clean cell content
                cell = cell.replace("|", "\\|").replace("\n", " ")
                cells.append(cell)
            row_str = "| " + " | ".join(cells) + " |"
            lines.append(row_str)
        
        markdown = "\n".join(lines)
        
        # Generate description
        description = self._generate_description(headers, data, title)
        
        return ExtractedTable(
            markdown=markdown,
            description=description,
            title=title,
            rows=num_rows,
            columns=num_cols,
            metadata={
                "headers": headers,
                "type": "table"
            }
        )
    
    def _generate_description(
        self, 
        headers: List[str], 
        data: List[List[Any]], 
        title: Optional[str]
    ) -> str:
        """Generate a natural language description of the table."""
        parts = []
        
        if title:
            parts.append(f"Table: {title}.")
        
        parts.append(f"This table has {len(data)} rows and {len(headers)} columns.")
        parts.append(f"Columns: {', '.join(headers)}.")
        
        # Add sample data description
        if data:
            first_col_vals = [str(row[0]) for row in data[:3] if row]
            if first_col_vals:
                parts.append(f"First column values include: {', '.join(first_col_vals)}.")
        
        return " ".join(parts)
    
    def _find_markdown_tables(self, text: str) -> List[ExtractedTable]:
        """Find markdown-formatted tables in text."""
        tables = []
        
        # Pattern for markdown table (header + separator + rows)
        pattern = r'(\|[^\n]+\|\n\|[-:\|\s]+\|\n(?:\|[^\n]+\|\n?)+)'
        
        matches = re.finditer(pattern, text)
        for match in matches:
            table_text = match.group(1)
            lines = [l.strip() for l in table_text.strip().split("\n")]
            
            if len(lines) >= 3:
                # Parse header
                header_line = lines[0]
                headers = [c.strip() for c in header_line.split("|") if c.strip()]
                
                # Parse data rows (skip separator)
                data = []
                for line in lines[2:]:
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    if cells:
                        data.append(cells)
                
                tables.append(ExtractedTable(
                    markdown=table_text,
                    description=self._generate_description(headers, data, None),
                    title=None,
                    rows=len(data),
                    columns=len(headers),
                    metadata={"headers": headers}
                ))
        
        return tables
    
    def _find_tsv_tables(self, text: str) -> List[ExtractedTable]:
        """Find tab-separated tables in text."""
        tables = []
        
        lines = text.split("\n")
        current_table = []
        
        for line in lines:
            if "\t" in line:
                cells = line.split("\t")
                if len(cells) >= 2:
                    current_table.append(cells)
            else:
                if len(current_table) >= 2:
                    # Convert accumulated TSV data to table
                    headers = current_table[0]
                    data = current_table[1:]
                    table = self.convert_to_markdown(data, headers)
                    tables.append(table)
                current_table = []
        
        # Handle last table
        if len(current_table) >= 2:
            headers = current_table[0]
            data = current_table[1:]
            table = self.convert_to_markdown(data, headers)
            tables.append(table)
        
        return tables
    
    def format_as_context(self, table: ExtractedTable) -> str:
        """
        Format table as context for RAG retrieval.
        Combines description and markdown representation.
        
        Args:
            table: ExtractedTable to format
            
        Returns:
            Formatted string for chunking
        """
        parts = []
        
        if table.title:
            parts.append(f"## {table.title}")
        
        parts.append(table.description)
        parts.append("")
        parts.append(table.markdown)
        
        if table.context:
            parts.append("")
            parts.append(f"Context: {table.context}")
        
        return "\n".join(parts)
