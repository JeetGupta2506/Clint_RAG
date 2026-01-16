"""
ChromaDB Manager for vector storage operations.
Handles collections, indexing, and retrieval.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
import chromadb

from .embeddings import EmbeddingsManager


@dataclass
class SearchResult:
    """Result from a vector search."""
    content: str
    chunk_id: str
    score: float
    metadata: Dict[str, Any]


class ChromaManager:
    """
    Manages ChromaDB collections and operations.
    Supports multiple collections for different websites/sources.
    """
    
    # Default collection for internal Daruka documents
    DARUKA_COLLECTION = "daruka_documents"
    
    def __init__(
        self, 
        persist_directory: str,
        embeddings_manager: EmbeddingsManager
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            persist_directory: Path to ChromaDB storage
            embeddings_manager: EmbeddingsManager instance
        """
        self.persist_directory = persist_directory
        self.embeddings_manager = embeddings_manager
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with new API
        self._client = chromadb.PersistentClient(path=persist_directory)
        
        # Cache for collections
        self._collections: Dict[str, Any] = {}
    
    def get_or_create_collection(self, name: str) -> Any:
        """
        Get or create a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection
        """
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        collection_name: str = None
    ) -> int:
        """
        Add documents to a collection.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: List of unique IDs
            collection_name: Target collection (defaults to DARUKA_COLLECTION)
            
        Returns:
            Number of documents added
        """
        collection_name = collection_name or self.DARUKA_COLLECTION
        collection = self.get_or_create_collection(collection_name)
        
        # Generate embeddings
        embeddings = self.embeddings_manager.embed_texts(documents)
        
        # Clean metadata - ChromaDB only accepts str, int, float, bool
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    clean_meta[k] = ""
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)
        
        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=clean_metadatas,
            ids=ids
        )
        
        return len(documents)
    
    def search(
        self,
        query: str,
        collection_names: List[str] = None,
        top_k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents across collections.
        
        Args:
            query: Search query
            collection_names: Collections to search (None = search all)
            top_k: Number of results per collection
            filter_dict: Metadata filter
            
        Returns:
            List of SearchResult objects
        """
        if collection_names is None:
            collection_names = self.list_collections()
        
        if not collection_names:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings_manager.embed_text(query)
        
        all_results = []
        
        for coll_name in collection_names:
            try:
                collection = self.get_or_create_collection(coll_name)
                
                # Build query params
                query_params = {
                    "query_embeddings": [query_embedding],
                    "n_results": top_k,
                    "include": ["documents", "metadatas", "distances"]
                }
                
                if filter_dict:
                    query_params["where"] = filter_dict
                
                results = collection.query(**query_params)
                
                # Parse results
                if results and results["documents"]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0]
                    distances = results["distances"][0]
                    ids = results["ids"][0]
                    
                    for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
                        # Convert distance to similarity score
                        score = 1 - dist  # Cosine distance to similarity
                        
                        all_results.append(SearchResult(
                            content=doc,
                            chunk_id=chunk_id,
                            score=score,
                            metadata={**meta, "collection": coll_name}
                        ))
                        
            except Exception as e:
                print(f"Search error in collection {coll_name}: {e}")
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def list_collections(self) -> List[str]:
        """Get list of all collection names."""
        collections = self._client.list_collections()
        return [c.name for c in collections]
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Dict with collection stats
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            count = collection.count()
            return {
                "name": collection_name,
                "count": count
            }
        except Exception as e:
            return {"name": collection_name, "count": 0, "error": str(e)}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all collections.
        
        Returns:
            Dict with overall stats
        """
        collections = self.list_collections()
        total_chunks = 0
        collection_stats = {}
        
        for name in collections:
            stats = self.get_collection_stats(name)
            collection_stats[name] = stats["count"]
            total_chunks += stats["count"]
        
        return {
            "total_collections": len(collections),
            "total_chunks": total_chunks,
            "collections": collections,
            "chunks_per_collection": collection_stats
        }
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if successful
        """
        try:
            self._client.delete_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            return True
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def clear_all(self) -> List[str]:
        """
        Clear all collections.
        
        Returns:
            List of cleared collection names
        """
        cleared = []
        for name in self.list_collections():
            if self.delete_collection(name):
                cleared.append(name)
        return cleared
    
    def persist(self):
        """Persist the database to disk. (Not needed with PersistentClient)"""
        pass  # PersistentClient auto-persists
