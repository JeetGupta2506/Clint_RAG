"""
Simple PDF Processor for extracting text from PDF documents.
Uses PyPDF2 for basic text extraction.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from PyPDF2 import PdfReader


@dataclass
class ExtractedPage:
    """Represents extracted content from a PDF page."""
    page_number: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFProcessor:
    """
    Simple processor for extracting text from PDF documents.
    """
    
    def __init__(self, **kwargs):
        """Initialize PDF processor."""
        pass
    
    def process(self, file_path: str) -> List[ExtractedPage]:
        """
        Extract text from all pages of a PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of ExtractedPage objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        pages = []
        filename = os.path.basename(file_path)
        
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            
            metadata = {
                "source": filename,
                "page": page_num,
                "total_pages": total_pages,
                "type": "pdf"
            }
            
            pages.append(ExtractedPage(
                page_number=page_num,
                content=text,
                metadata=metadata
            ))
        
        return pages
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract all text from a PDF as a single string.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Combined text from all pages
        """
        pages = self.process(file_path)
        return "\n\n".join(page.content for page in pages if page.content)
