"""
Session management endpoints for conversation memory.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag import get_memory


router = APIRouter()


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    website_context: str
    message_count: int


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: list
    total: int


@router.get("/sessions")
async def list_sessions(website_context: Optional[str] = None):
    """
    List all active conversation sessions.
    
    - Optionally filter by website_context
    """
    memory = get_memory()
    sessions = memory.list_sessions(website_context)
    
    return {
        "sessions": sessions,
        "total": len(sessions),
        "website_filter": website_context
    }


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, website_context: str = "default"):
    """
    Get information about a specific session.
    """
    memory = get_memory()
    conversation = memory.get_or_create_session(session_id, website_context)
    
    return {
        "session_id": session_id,
        "website_context": website_context,
        "message_count": len(conversation.messages),
        "created_at": conversation.created_at.isoformat(),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in conversation.messages
        ]
    }


@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str, 
    website_context: str = "default",
    confirm: bool = False
):
    """
    Clear a specific conversation session.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm session deletion"
        )
    
    memory = get_memory()
    memory.clear_session(session_id, website_context)
    
    return {
        "message": f"Cleared session '{session_id}' for website '{website_context}'",
        "session_id": session_id,
        "website_context": website_context
    }


@router.delete("/sessions")
async def clear_all_sessions(
    website_context: Optional[str] = None,
    confirm: bool = False
):
    """
    Clear all sessions, optionally filtered by website_context.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm deletion of all sessions"
        )
    
    memory = get_memory()
    
    if website_context:
        memory.clear_website_sessions(website_context)
        return {
            "message": f"Cleared all sessions for website '{website_context}'",
            "website_context": website_context
        }
    else:
        # Clear all
        memory._conversations.clear()
        return {
            "message": "Cleared all conversation sessions"
        }
