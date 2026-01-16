"""
Q&A Pair Chunker for FAQ and question-answer formatted content.
Keeps question-answer pairs together as single chunks.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import re


@dataclass
class QAChunk:
    """Represents a Q&A pair as a chunk."""
    content: str
    question: str
    answer: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QAChunker:
    """
    Chunker for Q&A pairs.
    Combines questions with their answers and preserves as single units.
    """
    
    def __init__(self):
        """Initialize Q&A chunker."""
        self.qa_patterns = [
            # Q: ... A: ... format
            r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)',
            # Question: ... Answer: ... format
            r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=Question:|$)',
            # **Question** ... **Answer** format
            r'\*\*Question\*\*:\s*(.+?)\s*\*\*Answer\*\*:\s*(.+?)(?=\*\*Question\*\*|$)',
        ]
    
    def chunk(
        self,
        qa_pairs: List[Tuple[str, str]],
        source: str = "unknown",
        website: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> List[QAChunk]:
        """
        Create chunks from Q&A pairs.
        
        Args:
            qa_pairs: List of (question, answer) tuples
            source: Source identifier
            website: Website source name
            base_metadata: Additional metadata
            
        Returns:
            List of QAChunk objects
        """
        base_metadata = base_metadata or {}
        chunks = []
        
        for i, (question, answer) in enumerate(qa_pairs):
            if not question.strip() or not answer.strip():
                continue
            
            # Format as Q&A content
            content = f"Q: {question.strip()}\nA: {answer.strip()}"
            
            chunk_id = f"{source}_qa_{i}"
            
            metadata = {
                **base_metadata,
                "source": source,
                "chunk_type": "qa",
                "question": question.strip(),
                "chunk_index": i,
            }
            
            if website:
                metadata["website"] = website
            
            chunks.append(QAChunk(
                content=content,
                question=question.strip(),
                answer=answer.strip(),
                chunk_id=chunk_id,
                metadata=metadata
            ))
        
        return chunks
    
    def chunk_from_text(
        self,
        text: str,
        source: str = "unknown",
        website: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> List[QAChunk]:
        """
        Extract and chunk Q&A pairs from text.
        
        Args:
            text: Text containing Q&A pairs
            source: Source identifier
            website: Website source name
            base_metadata: Additional metadata
            
        Returns:
            List of QAChunk objects
        """
        qa_pairs = self._extract_qa_pairs(text)
        return self.chunk(
            qa_pairs=qa_pairs,
            source=source,
            website=website,
            base_metadata=base_metadata
        )
    
    def chunk_from_rows(
        self,
        rows: List[Dict[str, Any]],
        question_key: str = None,
        answer_key: str = None,
        source: str = "unknown",
        website: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> List[QAChunk]:
        """
        Create chunks from row data (e.g., from spreadsheets).
        
        Args:
            rows: List of row dictionaries
            question_key: Key for question column (auto-detected if None)
            answer_key: Key for answer column (auto-detected if None)
            source: Source identifier
            website: Website source name
            base_metadata: Additional metadata
            
        Returns:
            List of QAChunk objects
        """
        if not rows:
            return []
        
        # Auto-detect Q&A columns if not specified
        if question_key is None or answer_key is None:
            q_key, a_key = self._detect_qa_columns(rows[0].keys())
            question_key = question_key or q_key
            answer_key = answer_key or a_key
        
        if not question_key or not answer_key:
            return []
        
        qa_pairs = []
        for row in rows:
            question = str(row.get(question_key, "")).strip()
            answer = str(row.get(answer_key, "")).strip()
            if question and answer:
                qa_pairs.append((question, answer))
        
        return self.chunk(
            qa_pairs=qa_pairs,
            source=source,
            website=website,
            base_metadata=base_metadata
        )
    
    def _extract_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Extract Q&A pairs from text using patterns."""
        pairs = []
        
        for pattern in self.qa_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    pairs.append((match[0].strip(), match[1].strip()))
        
        return pairs
    
    def _detect_qa_columns(self, columns) -> Tuple[str, str]:
        """Auto-detect question and answer columns."""
        columns = list(columns)
        col_lower = {c: c.lower() for c in columns}
        
        question_key = None
        answer_key = None
        
        q_indicators = ["question", "q", "query", "faq"]
        a_indicators = ["answer", "a", "response", "reply"]
        
        for col, lower in col_lower.items():
            for q in q_indicators:
                if q in lower:
                    question_key = col
                    break
            for a in a_indicators:
                if a in lower:
                    answer_key = col
                    break
        
        return question_key, answer_key
