"""
RAG Chain for query processing and answer generation with memory support.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .retriever import RAGRetriever, RetrievedDocument
from .prompts import PromptTemplates
from .memory import get_memory


@dataclass
class RAGResponse:
    """Response from RAG chain."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    documents_used: int
    session_id: Optional[str] = None


class RAGChain:
    """
    RAG Chain combining retriever, prompts, LLM, and conversation memory.
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
        self.memory = get_memory()
    
    def query(
        self,
        question: str,
        website_context: Optional[str] = None,
        top_k: int = 5,
        session_id: Optional[str] = None
    ) -> RAGResponse:
        """
        Process a query through the RAG chain with optional memory.
        
        Args:
            question: User question
            website_context: Optional website filter
            top_k: Number of documents to retrieve
            session_id: Optional session ID for conversation memory
            
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
        
        # Get conversation history if session_id provided
        conversation_history = ""
        if session_id:
            conversation_history = self.memory.get_formatted_history(
                session_id=session_id,
                website_context=website_context or "default",
                max_messages=6
            )
        
        # Get prompts with memory
        system_prompt, user_prompt = self.prompts.get_full_prompt(
            context=context,
            question=question,
            conversation_history=conversation_history
        )
        
        # Generate answer
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Save to memory if session_id provided
        if session_id:
            self.memory.add_exchange(
                session_id=session_id,
                website_context=website_context or "default",
                user_message=question,
                assistant_message=answer
            )
        
        # Format sources
        sources = self.retriever.get_sources_for_response(documents)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            documents_used=len(documents),
            session_id=session_id
        )
    
    def query_with_custom_context(
        self,
        question: str,
        context: str,
        session_id: Optional[str] = None,
        website_context: Optional[str] = None
    ) -> str:
        """
        Generate answer with custom context (bypass retrieval).
        
        Args:
            question: User question
            context: Pre-formatted context
            session_id: Optional session ID for memory
            website_context: Optional website context for memory
            
        Returns:
            Generated answer
        """
        # Get conversation history if session_id provided
        conversation_history = ""
        if session_id:
            conversation_history = self.memory.get_formatted_history(
                session_id=session_id,
                website_context=website_context or "default",
                max_messages=6
            )
        
        system_prompt, user_prompt = self.prompts.get_full_prompt(
            context=context,
            question=question,
            conversation_history=conversation_history
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Save to memory if session_id provided
        if session_id:
            self.memory.add_exchange(
                session_id=session_id,
                website_context=website_context or "default",
                user_message=question,
                assistant_message=answer
            )
        
        return answer
