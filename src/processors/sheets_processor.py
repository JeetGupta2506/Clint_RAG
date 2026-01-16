"""
Google Sheets Processor for fetching and processing data from Google Sheets.
Uses gspread for API access.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SheetRow:
    """Represents a row from a Google Sheet."""
    row_number: int
    data: Dict[str, Any]
    sheet_name: str
    is_qa_pair: bool = False
    is_table_row: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SheetData:
    """Represents data from a worksheet."""
    sheet_name: str
    rows: List[SheetRow]
    headers: List[str]
    is_qa_format: bool = False
    is_table_format: bool = False
    raw_data: List[List[Any]] = field(default_factory=list)


class SheetsProcessor:
    """
    Processor for fetching and processing Google Sheets data.
    Each worksheet represents data from a different website (via n8n).
    """
    
    def __init__(self, credentials_path: str):
        """
        Initialize Sheets processor.
        
        Args:
            credentials_path: Path to Google service account credentials JSON
        """
        self.credentials_path = credentials_path
        self._client = None
        self._initialized = False
    
    def _init_client(self):
        """Initialize gspread client with credentials."""
        if self._initialized:
            return
        
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Google credentials not found: {self.credentials_path}"
            )
        
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials
            
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_path, scope
            )
            self._client = gspread.authorize(credentials)
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Sheets client: {e}")
    
    def get_all_worksheets(self, sheet_id: str) -> List[str]:
        """
        Get all worksheet names from a Google Sheet.
        
        Args:
            sheet_id: Google Sheet ID
            
        Returns:
            List of worksheet names
        """
        self._init_client()
        
        try:
            spreadsheet = self._client.open_by_key(sheet_id)
            return [ws.title for ws in spreadsheet.worksheets()]
        except Exception as e:
            raise RuntimeError(f"Failed to get worksheets: {e}")
    
    def process_sheet(self, sheet_id: str, worksheet_name: Optional[str] = None) -> List[SheetData]:
        """
        Process all worksheets or a specific worksheet from a Google Sheet.
        
        Args:
            sheet_id: Google Sheet ID
            worksheet_name: Optional specific worksheet to process
            
        Returns:
            List of SheetData objects
        """
        self._init_client()
        
        try:
            spreadsheet = self._client.open_by_key(sheet_id)
            worksheets = (
                [spreadsheet.worksheet(worksheet_name)] 
                if worksheet_name 
                else spreadsheet.worksheets()
            )
            
            results = []
            for ws in worksheets:
                sheet_data = self._process_worksheet(ws)
                if sheet_data:
                    results.append(sheet_data)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to process sheet: {e}")
    
    def _process_worksheet(self, worksheet) -> Optional[SheetData]:
        """Process a single worksheet."""
        try:
            all_values = worksheet.get_all_values()
            
            if not all_values or len(all_values) < 2:
                return None
            
            headers = all_values[0]
            data_rows = all_values[1:]
            
            # Detect format type
            is_qa_format = self._is_qa_format(headers)
            is_table_format = self._is_table_format(headers, data_rows)
            
            rows = []
            for i, row_data in enumerate(data_rows, start=2):
                # Convert row to dict with headers as keys
                row_dict = {
                    headers[j]: row_data[j] if j < len(row_data) else ""
                    for j in range(len(headers))
                }
                
                # Skip empty rows
                if not any(v.strip() for v in row_dict.values() if isinstance(v, str)):
                    continue
                
                rows.append(SheetRow(
                    row_number=i,
                    data=row_dict,
                    sheet_name=worksheet.title,
                    is_qa_pair=is_qa_format,
                    is_table_row=is_table_format,
                    metadata={
                        "website": worksheet.title,
                        "source": f"Sheet: {worksheet.title}",
                        "type": "sheets"
                    }
                ))
            
            return SheetData(
                sheet_name=worksheet.title,
                rows=rows,
                headers=headers,
                is_qa_format=is_qa_format,
                is_table_format=is_table_format,
                raw_data=all_values
            )
            
        except Exception as e:
            print(f"Error processing worksheet {worksheet.title}: {e}")
            return None
    
    def _is_qa_format(self, headers: List[str]) -> bool:
        """Check if headers indicate Q&A format."""
        header_lower = [h.lower() for h in headers]
        qa_indicators = ["question", "answer", "q", "a", "faq", "query", "response"]
        return any(indicator in h for h in header_lower for indicator in qa_indicators)
    
    def _is_table_format(self, headers: List[str], data: List[List[Any]]) -> bool:
        """Check if data is in table format (structured with multiple columns)."""
        # Consider it a table if it has more than 2 columns with data
        if len(headers) <= 2:
            return False
        
        # Check if most rows have data in multiple columns
        multi_col_rows = 0
        for row in data[:10]:  # Sample first 10 rows
            non_empty = sum(1 for cell in row if str(cell).strip())
            if non_empty >= 3:
                multi_col_rows += 1
        
        return multi_col_rows >= len(data[:10]) * 0.5
