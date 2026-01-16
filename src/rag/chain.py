"""
RAG Chain for query processing and answer generation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .retriever import RAGRetriever, RetrievedDocument
from .prompts import PromptTemplates


@dataclass
class RAGResponse:
    """Response from RAG chain."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    documents_used: int


class RAGChain:
    """
    RAG Chain combining retriever, prompts, and LLM.
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.0
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: RAGRetriever instance
            api_key: Anthropic API key
            model: Claude model name
            temperature: Generation temperature
        """
        self.retriever = retriever
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model,
            temperature=temperature
        )
        self.prompts = PromptTemplates()
    
    def query(
        self,
        question: str,
        website_context: Optional[str] = None,
        top_k: int = 5
    ) -> RAGResponse:
        """
        Process a query through the RAG chain.
        
        Args:
            question: User question
            website_context: Optional website filter
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve relevant documents
        documents = self.retriever.retrieve(
            query=question,
            website_context=website_context,
            top_k=top_k
        )
        
        # Format context
        context = self.retriever.format_context(documents)
        
        # Get prompts
        system_prompt, user_prompt = self.prompts.get_full_prompt(
            context=context,
            question=question
        )
        
        # Generate answer
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Format sources
        sources = self.retriever.get_sources_for_response(documents)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            documents_used=len(documents)
        )
    
    def query_with_custom_context(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Generate answer with custom context (bypass retrieval).
        
        Args:
            question: User question
            context: Pre-formatted context
            
        Returns:
            Generated answer
        """
        system_prompt, user_prompt = self.prompts.get_full_prompt(
            context=context,
            question=question
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
