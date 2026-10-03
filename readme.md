# 📚 Production RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot that lets users upload multiple PDF documents and have grounded, conversational Q&A over them — with hybrid search, streaming responses, and source citations.

Built as a full-stack system: a **FastAPI backend** handling ingestion, retrieval, and generation, paired with a **Streamlit frontend** for interactive chat.

---

## ✨ Features

- **Multi-document support** — upload, list, and delete PDFs independently, each tracked with a unique ID (no filename collisions)
- **Persistent vector storage** — ChromaDB-backed, survives restarts
- **Hybrid retrieval** — combines semantic search (embeddings) with keyword search (BM25) via Reciprocal Rank Fusion, for better recall on both conceptual and exact-term queries
- **Conversational memory** — follow-up questions are automatically rewritten with context from the chat history (e.g. "what's its solution?" → "what is the solution to overfitting?")
- **Scoped or global search** — query across all uploaded documents, or restrict to a single one
- **Streaming responses** — answers are streamed token-by-token over Server-Sent Events (SSE), not returned all at once
- **Source citations** — every answer includes the source filenames it was grounded in
- **Structured logging & error handling** — request-ID tracing across logs, centralized exception handling, daily log rotation
- **Rate-limit resilient ingestion** — automatic batching and retry logic for embedding large documents

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI |
| LLM orchestration | LangChain (LCEL) |
| LLM & Embeddings | Google Gemini |
| Vector store | ChromaDB |
| Keyword search | BM25 (`rank_bm25`) |
| Frontend | Streamlit |
| Config management | Pydantic Settings |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
production_rag/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point, middleware, exception handlers
│   │   ├── core/
│   │   │   ├── config.py            # Centralized settings (env-driven)
│   │   │   ├── logger.py            # Logging setup (daily rotation, request-ID injection)
│   │   │   ├── middleware.py        # Request logging middleware
│   │   │   ├── request_context.py   # Request-ID context management
│   │   │   ├── embeddings.py        # Gemini embeddings wrapper (with retry)
│   │   │   ├── vectorstore.py       # ChromaDB client, batched ingestion
│   │   │   ├── document_processor.py # PDF loading, chunking, ID tagging
│   │   │   └── chains.py            # Hybrid retriever + conversational RAG chain
│   │   └── api/routes/
│   │       ├── health.py
│   │       ├── upload.py
│   │       ├── documents.py
│   │       └── chat.py              # /chat and /chat/stream endpoints
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### 1. Backend setup

```bash
cd production_rag
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

cd backend
pip install -r requirements.txt
```

Create a `.env` file in the project root (`production_rag/.env`):

```env
GOOGLE_API_KEY=your_key_here
```

Run the backend:

```bash
uvicorn app.main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`.

### 2. Frontend setup

In a **separate terminal**, using a **separate virtual environment** (kept isolated from the backend to avoid dependency conflicts):

```bash
cd production_rag
python -m venv venv-frontend
venv-frontend\Scripts\activate

cd frontend
pip install -r requirements.txt
```

Create a `.env` file inside `frontend/`:

```env
BACKEND_URL=http://127.0.0.1:8000
```

Run the frontend:

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## ⚙️ Environment Variables

| Variable | Location | Description | Default |
|---|---|---|---|
| `GOOGLE_API_KEY` | `backend/.env` | Gemini API key (required) | — |
| `CHROMA_PERSIST_DIR` | `backend/.env` | ChromaDB storage path | `chroma_db` |
| `CHUNK_SIZE` | `backend/.env` | Text chunk size for splitting | `1000` |
| `CHUNK_OVERLAP` | `backend/.env` | Overlap between chunks | `200` |
| `LLM_MODEL` | `backend/.env` | Gemini chat model | `gemini-3.6-flash` |
| `EMBEDDING_MODEL` | `backend/.env` | Gemini embedding model | `models/gemini-embedding-2` |
| `RETRIEVAL_K` | `backend/.env` | Chunks retrieved per query | `10` |
| `BACKEND_URL` | `frontend/.env` | Backend API base URL | `http://127.0.0.1:8000` |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health/` | Health check |
| `POST` | `/upload/` | Upload and ingest a PDF |
| `GET` | `/documents/` | List all uploaded documents |
| `DELETE` | `/documents/{document_id}` | Delete a document and its chunks |
| `POST` | `/chat/` | Ask a question (full response) |
| `POST` | `/chat/stream` | Ask a question (streamed response, SSE) |

Full interactive documentation available via Swagger UI at `/docs`.

---

## 💬 Usage

1. Start both the backend and frontend (see above)
2. Upload one or more PDFs from the sidebar
3. Optionally select a specific document to scope your search, or leave it on "All documents"
4. Ask questions in the chat box — responses stream in real time, with source citations shown below each answer
5. Ask follow-up questions naturally — the system maintains conversational context per session

---

## 👤 Author

Built as a hands-on learning project to explore production-grade RAG system design — from ingestion and retrieval to conversational memory, streaming, and observability.