"""
Embeddings Manager using HuggingFace sentence-transformers (FREE, runs locally).
"""

from typing import List
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingsManager:
    """
    Manages embedding generation using HuggingFace sentence-transformers.
    Runs locally - no API key required!
    """
    
    # Default model - fast, good quality, 384 dimensions
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, model: str = None):
        """
        Initialize embeddings manager.
        
        Args:
            model: HuggingFace model name (defaults to all-MiniLM-L6-v2)
        """
        self.model = model or self.DEFAULT_MODEL
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"✅ Loaded embedding model: {self.model}")
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get the embeddings instance for LangChain."""
        return self._embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self._embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model."""
        dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        }
        return dimensions.get(self.model, 384)


@lru_cache(maxsize=1)
def get_embeddings_manager(model: str = None) -> EmbeddingsManager:
    """Get cached embeddings manager instance."""
    return EmbeddingsManager(model=model)
