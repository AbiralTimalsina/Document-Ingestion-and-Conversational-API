# Document Ingestion & Conversational RAG API

A FastAPI backend with **two REST APIs**: a **Document Ingestion API** for uploading, chunking, embedding, and storing documents; and a **Conversational RAG API** with custom retrieval-augmented generation, Redis-backed multi-turn chat memory, and LLM-driven interview booking.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | **FastAPI** (v0.141+) |
| Vector Database | **Qdrant**  |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim) |
| SQL Database | **SQLite** + **SQLAlchemy** ORM |
| PDF Parsing | **PyMuPDF** |
| LLM | **OpenAI** (`gpt-4o-mini`) |
| Chat Memory | **Redis** (session-keyed lists with TTL) |

---

## Quick Start

### 1. Clone & install

```bash
git clone [<repo-url>](https://github.com/AbiralTimalsina/Document-Ingestion-and-Conversational-API)
cd "Document Ingestion API"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Redis

```bash
# macOS
brew install redis
redis-server --daemonize yes

# Verify
redis-cli ping   # → PONG
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-key-here
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

> **First run:** The embedding model (`all-MiniLM-L6-v2`, ~80 MB) downloads automatically.

### 5. Interactive docs

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---



## API Reference

### Health Check

```
GET /
```

Returns service status for database, Qdrant, Redis, embedding model, and LLM.

---

### API 1: Document Ingestion

#### Upload a document

```
POST /api/v1/documents/upload
```

| Parameter | Type | Description |
|---|---|---|
| `file` | `UploadFile` | PDF or TXT file (max 50 MB) |
| `chunking_strategy` | `string` | `fixed_size` or `recursive` (default: `recursive`) |
| `chunk_size` | `int` | Characters per chunk (optional, default: 512) |
| `chunk_overlap` | `int` | Overlap characters (optional, default: 50) |

**Pipeline:** Upload → Extract text → Chunk → Embed → Store in Qdrant → Save metadata in SQLite

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@myfile.pdf" \
  -F "chunking_strategy=recursive"
```

#### List documents

```
GET /api/v1/documents/?page=1&page_size=10
```

#### Get document details (with chunks)

```
GET /api/v1/documents/{document_id}
```

#### Get document chunks

```
GET /api/v1/documents/{document_id}/chunks?page=1&page_size=20
```

#### Delete a document

```
DELETE /api/v1/documents/{document_id}
```

Removes the document, all chunks from SQLite, all vectors from Qdrant, and the uploaded file from disk.

#### Semantic search

```
POST /api/v1/search/
```

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "what is deep learning?", "top_k": 5}'
```

Optionally filter by document: `"document_id": "YOUR_DOC_ID"`

---

### API 2: Conversational RAG

#### Send a chat message

```
POST /api/v1/chat/
```

| Field | Type | Description |
|---|---|---|
| `session_id` | `string` | Unique session ID for conversation continuity |
| `message` | `string` | User message (max 5000 chars) |
| `document_id` | `string?` | Optional: limit retrieval to a specific document |

```bash
# First message
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-1", "message": "What is machine learning?"}'

# Follow-up (multi-turn — uses conversation context)
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-1", "message": "How does it differ from deep learning?"}'
```

**Response includes:**
- `response` — LLM-generated answer grounded in document context
- `sources` — Retrieved chunks with document names and relevance scores
- `booking` — Extracted booking data (if booking intent detected)
- `booking_id` — Database ID of created booking (if applicable)

#### View chat history

```
GET /api/v1/chat/{session_id}/history
```

#### Clear a chat session

```
DELETE /api/v1/chat/{session_id}
```

#### Interview booking via chat

The LLM conversationally collects four fields: **name**, **email**, **date**, **time**. When all four are present, the booking is automatically extracted and stored.

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "booking-1", "message": "I want to book an interview. My name is John Smith, email john@example.com, for Monday Aug 18 at 2:30 PM"}'
```

#### List bookings

```
GET /api/v1/bookings/?session_id=booking-1
```

#### Get a specific booking

```
GET /api/v1/bookings/{booking_id}
```

---

## Chunking Strategies

### 1. `fixed_size`

Splits text into chunks of N characters with M overlap. Simple and predictable, but may cut words mid-sentence.

### 2. `recursive`

Recursively splits by paragraph breaks (`\n\n`) → line breaks (`\n`) → sentences (`. `) → spaces (` `). Preserves semantic coherence and natural text boundaries. **Recommended for most use cases.**

---


### Multi-Turn Handling

The query rewriter uses the last 3 conversation turns to expand vague follow-ups into standalone queries. For example:

| Turn | User Says | Rewritten Query |
|---|---|---|
| 1 | "What is machine learning?" | "What is machine learning?" |
| 2 | "How is it different from AI?" | "How is machine learning different from artificial intelligence?" |
| 3 | "Give me examples" | "Give me examples of machine learning applications" |

### Booking Flow

The system prompt instructs the LLM to collect booking fields one at a time in natural conversation. When all four fields are detected across the conversation, a second focused LLM call extracts structured data and stores it.

---

## Project Structure

```
Document Ingestion API/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, health check, CORS
│   ├── config.py                # Pydantic settings from .env
│   ├── database.py              # SQLAlchemy engine, session factory
│   ├── models.py                # ORM models: Document, Chunk, Booking
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── routers/
│   │   ├── documents.py         # Upload, list, get, delete endpoints
│   │   ├── search.py            # Semantic similarity search
│   │   └── chat.py              # Conversational RAG + booking endpoints
│   └── services/
│       ├── extractor.py         # PDF & TXT text extraction
│       ├── chunker.py           # Fixed-size & recursive chunking
│       ├── embedder.py          # Sentence-transformer embedding wrapper
│       ├── vector_store.py      # Qdrant client (upsert, search, delete)
│       ├── chat_memory.py       # Redis chat history manager
│       ├── llm.py               # OpenAI API wrapper (completions + extraction)
│       └── rag_pipeline.py      # Custom RAG orchestrator (5-step pipeline)
├── requirements.txt
├── .env.example
└── README.md
```

