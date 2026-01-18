"""
Conversation Memory Manager for maintaining chat context per website.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Conversation:
    """A conversation session for a specific website context."""
    session_id: str
    website_context: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))
    
    def get_history(self, max_messages: int = 10) -> List[Message]:
        """Get recent conversation history."""
        return self.messages[-max_messages:]
    
    def format_for_prompt(self, max_messages: int = 6) -> str:
        """Format conversation history for inclusion in prompt."""
        history = self.get_history(max_messages)
        if not history:
            return ""
        
        formatted = "\n=== CONVERSATION HISTORY ===\n"
        for msg in history:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted += f"{role_label}: {msg.content}\n\n"
        formatted += "=== END HISTORY ===\n"
        return formatted
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "website_context": self.website_context,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat()
        }


class ConversationMemory:
    """
    Manages conversation memory across multiple sessions.
    Each website context can have multiple sessions.
    """
    
    def __init__(self, max_sessions_per_website: int = 100):
        # Structure: {website_context: {session_id: Conversation}}
        self._conversations: Dict[str, Dict[str, Conversation]] = {}
        self.max_sessions_per_website = max_sessions_per_website
    
    def get_or_create_session(
        self, 
        session_id: str, 
        website_context: str = "default"
    ) -> Conversation:
        """Get existing session or create a new one."""
        website_context = website_context or "default"
        
        if website_context not in self._conversations:
            self._conversations[website_context] = {}
        
        if session_id not in self._conversations[website_context]:
            # Create new conversation
            self._conversations[website_context][session_id] = Conversation(
                session_id=session_id,
                website_context=website_context
            )
            
            # Cleanup old sessions if too many
            self._cleanup_old_sessions(website_context)
        
        return self._conversations[website_context][session_id]
    
    def add_exchange(
        self,
        session_id: str,
        website_context: str,
        user_message: str,
        assistant_message: str
    ):
        """Add a user-assistant exchange to the conversation."""
        conversation = self.get_or_create_session(session_id, website_context)
        conversation.add_message("user", user_message)
        conversation.add_message("assistant", assistant_message)
    
    def get_formatted_history(
        self,
        session_id: str,
        website_context: str = "default",
        max_messages: int = 6
    ) -> str:
        """Get formatted conversation history for prompt."""
        conversation = self.get_or_create_session(session_id, website_context)
        return conversation.format_for_prompt(max_messages)
    
    def clear_session(self, session_id: str, website_context: str = "default"):
        """Clear a specific session."""
        website_context = website_context or "default"
        if website_context in self._conversations:
            if session_id in self._conversations[website_context]:
                del self._conversations[website_context][session_id]
    
    def clear_website_sessions(self, website_context: str):
        """Clear all sessions for a website context."""
        if website_context in self._conversations:
            del self._conversations[website_context]
    
    def get_session_info(self, session_id: str, website_context: str = "default") -> dict:
        """Get session information."""
        conversation = self.get_or_create_session(session_id, website_context)
        return {
            "session_id": session_id,
            "website_context": website_context,
            "message_count": len(conversation.messages),
            "created_at": conversation.created_at.isoformat()
        }
    
    def list_sessions(self, website_context: str = None) -> List[dict]:
        """List all active sessions."""
        sessions = []
        
        if website_context:
            if website_context in self._conversations:
                for session_id, conv in self._conversations[website_context].items():
                    sessions.append({
                        "session_id": session_id,
                        "website_context": website_context,
                        "message_count": len(conv.messages)
                    })
        else:
            for wc, convs in self._conversations.items():
                for session_id, conv in convs.items():
                    sessions.append({
                        "session_id": session_id,
                        "website_context": wc,
                        "message_count": len(conv.messages)
                    })
        
        return sessions
    
    def _cleanup_old_sessions(self, website_context: str):
        """Remove oldest sessions if over limit."""
        if website_context not in self._conversations:
            return
        
        sessions = self._conversations[website_context]
        if len(sessions) > self.max_sessions_per_website:
            # Sort by creation time and remove oldest
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: x[1].created_at
            )
            
            # Remove oldest sessions
            to_remove = len(sessions) - self.max_sessions_per_website
            for session_id, _ in sorted_sessions[:to_remove]:
                del sessions[session_id]


# Global memory instance
_memory_instance = None


def get_memory() -> ConversationMemory:
    """Get the global conversation memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance
