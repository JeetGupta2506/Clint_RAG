# Daruka.Earth RAG System

AI-powered RAG (Retrieval-Augmented Generation) system for Daruka.Earth with project matching and grant pitch capabilities.

## Features

- **PDF Upload & Processing** - Extract and chunk PDF documents
- **Text Ingestion** - Add website/company content directly
- **Multi-Collection Support** - Separate collections per data source
- **Conversation Memory** - Per-website session context
- **Project Matching** - Find existing projects or generate hypothetical ones
- **Grant Pitch Generation** - Context-aware pitch responses for grant applications
- **Claude LLM** - Haiku model for cost-effective responses
- **Free Embeddings** - HuggingFace sentence-transformers (local)

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

# 4. Seed sample projects (run once)
# POST http://localhost:8000/api/projects/seed
```

Open `http://localhost:8000/docs` for Swagger UI.

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload PDF documents |
| POST | `/api/ingest/text` | Add text content to collection |
| POST | `/api/query` | Query with RAG |
| POST | `/api/pitch` | **Generate grant pitch with project matching** |

### Project & Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects/seed` | Seed sample Daruka projects |
| GET | `/api/sessions` | List conversation sessions |
| DELETE | `/api/sessions/{id}?confirm=true` | Clear a session |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | View database stats |
| GET | `/api/collections` | List all collections |
| GET | `/api/collections/{name}` | View collection chunks |
| DELETE | `/api/collections/{name}?confirm=true` | Delete collection |

## Pitch Endpoint (Project Matching)

The `/api/pitch` endpoint is designed for grant applications:

```json
POST /api/pitch
{
  "grant_context": "rrcf_raptors",
  "question": "Describe your proposed project methodology",
  "grant_focus": "raptor conservation",
  "session_id": "grant_session_1"
}
```

**How it works:**
1. Searches `daruka_projects` collection for matching projects
2. If match found (score > 0.6), uses existing project
3. If no match, generates hypothetical project using LLM
4. Combines project + grant context for pitch response

**Response includes:**
- Pitch answer
- Project details (existing or generated)
- Source documents
- Session ID for follow-up questions

## Conversation Memory

Use `session_id` for multi-turn conversations:

```json
// First question
{ "query": "What grants are available?", "session_id": "user123" }

// Follow-up (same session_id)
{ "query": "What documents do I need?", "session_id": "user123" }
```

Each `website_context` has isolated memory.

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

## Sample Projects (Pre-seeded)

After running `/api/projects/seed`:

| Project | Focus | Status |
|---------|-------|--------|
| Sundarbans Biodiversity Credit | Mangroves, Carbon Credits | Active |
| BioGuardian Platform | AI Threat Detection, Climate | Development |
| Western Ghats Avian Monitoring | Birds, Raptors, Acoustic | Planned |
