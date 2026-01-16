"""
Ingest endpoint for Google Sheets synchronization.
Note: Google Sheets functionality requires additional setup.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/ingest/sheets")
async def ingest_google_sheets(sheet_id: str = None):
    """
    Sync data from Google Sheets.
    
    Note: This feature requires Google Sheets setup.
    """
    raise HTTPException(
        status_code=501,
        detail="Google Sheets integration not configured. Please set up credentials first."
    )
