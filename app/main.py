import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import Base, engine
from app.schemas import HealthResponse
from app.services import chat_memory, embedder, vector_store
from app.routers import chat, documents, search

 
# Logging
 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


 
# Lifespan (startup / shutdown)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""
    settings = get_settings()

    # 1. Create SQLite tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    # 2. Create upload directory
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # 3. Load embedding model (downloads on first run)
    logger.info("Loading embedding model...")
    embedder.load_model()

    # 4. Initialize Qdrant collection
    logger.info("Initializing Qdrant collection...")
    vector_store.ensure_collection()

    # 5. Check Redis connection
    logger.info("Checking Redis connection...")
    if chat_memory.ping():
        logger.info("Redis connected")
    else:
        logger.warning("Redis is not reachable — chat memory will not work")

    logger.info("All services initialized. API is ready!")
    yield

    # Shutdown
    logger.info("Shutting down...")


 
# FastAPI App
 

app = FastAPI(
    title="Document Ingestion and Conversational RAG API",
    description=(
        "Upload PDF/TXT documents, extract text, chunk with selectable strategies, "
        "generate embeddings, and store in Qdrant vector database with SQL metadata. "
        "Includes conversational RAG with Redis chat memory and interview booking."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
# Routers
 

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)


 
# Health check
 

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint with service status."""
    settings = get_settings()
    qdrant_info = vector_store.get_collection_info()
    redis_ok = chat_memory.ping()

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        services={
            "database": "connected",
            "vector_store": qdrant_info,
            "embedding_model": settings.EMBEDDING_MODEL,
            "redis": "connected" if redis_ok else "disconnected",
            "llm_model": settings.OPENAI_MODEL,
        },
    )
