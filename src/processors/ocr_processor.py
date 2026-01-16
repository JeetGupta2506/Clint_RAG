"""
OCR Processor for extracting text from scanned documents and images.
Uses pytesseract for OCR with pdf2image for PDF conversion.
"""

import os
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float = 0.0
    source: str = ""


class OCRProcessor:
    """
    OCR processor using pytesseract for text extraction from images.
    Supports PDF pages and image files.
    """
    
    def __init__(self, lang: str = "eng", tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR processor.
        
        Args:
            lang: Tesseract language code
            tesseract_cmd: Path to tesseract executable (optional)
        """
        self.lang = lang
        self._tesseract_available = False
        self._pdf2image_available = False
        
        # Check tesseract availability
        try:
            import pytesseract
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            # Test if tesseract is accessible
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception:
            print("⚠️ Tesseract not available. OCR functionality disabled.")
        
        # Check pdf2image availability
        try:
            import pdf2image
            self._pdf2image_available = True
        except ImportError:
            print("⚠️ pdf2image not available. PDF OCR functionality disabled.")
    
    @property
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self._tesseract_available
    
    def process_image(self, image_path: str) -> str:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text
        """
        if not self._tesseract_available:
            return ""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()
        except Exception as e:
            print(f"OCR error for image {image_path}: {e}")
            return ""
    
    def process_pdf_page(self, pdf_path: str, page_number: int) -> str:
        """
        Extract text from a specific PDF page using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            Extracted text from the page
        """
        if not self._tesseract_available or not self._pdf2image_available:
            return ""
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            # Convert specific page to image
            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                dpi=300  # Higher DPI for better OCR
            )
            
            if not images:
                return ""
            
            # OCR the page image
            text = pytesseract.image_to_string(images[0], lang=self.lang)
            return text.strip()
            
        except Exception as e:
            print(f"PDF OCR error for page {page_number}: {e}")
            return ""
    
    def process_pdf_all_pages(self, pdf_path: str) -> List[OCRResult]:
        """
        Extract text from all pages of a PDF using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of OCRResult objects for each page
        """
        if not self._tesseract_available or not self._pdf2image_available:
            return []
        
        results = []
        
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convert all pages to images
            images = convert_from_path(pdf_path, dpi=300)
            
            for i, image in enumerate(images, start=1):
                text = pytesseract.image_to_string(image, lang=self.lang)
                results.append(OCRResult(
                    text=text.strip(),
                    source=f"page_{i}"
                ))
                
        except Exception as e:
            print(f"PDF OCR error: {e}")
        
        return results
