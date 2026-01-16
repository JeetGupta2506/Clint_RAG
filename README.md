# Daruka.Earth RAG System

AI-powered RAG (Retrieval-Augmented Generation) system for Daruka.Earth using ChromaDB, Claude LLM, and free HuggingFace embeddings.

## Features

- **PDF Upload & Processing** - Extract and chunk PDF documents
- **Text Ingestion** - Add website/company content directly
- **Multi-Collection Support** - Separate collections per data source
- **Claude LLM** - Haiku model for cost-effective responses
- **Free Embeddings** - HuggingFace sentence-transformers (local)
- **Brand-Aligned Responses** - All answers framed from Daruka's perspective

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 3. Run
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for Swagger UI.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload PDF documents |
| POST | `/api/ingest/text` | Add text content to collection |
| POST | `/api/query` | Query with RAG |
| GET | `/api/stats` | View database stats |
| GET | `/api/collections` | List all collections |
| GET | `/api/collections/{name}` | View collection chunks |
| DELETE | `/api/collections/{name}?confirm=true` | Delete collection |

## Example Query

```json
POST /api/query
{
  "query": "What is Daruka's revenue model?",
  "top_k": 5
}
```

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: ChromaDB
- **LLM**: Claude 3.5 Haiku (Anthropic)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Chunking**: LangChain RecursiveCharacterTextSplitter

## Environment Variables

```env
ANTHROPIC_API_KEY=your-key-here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=claude-3-5-haiku-20241022
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```
