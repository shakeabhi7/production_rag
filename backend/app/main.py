from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown logic for the app.

    Code BEFORE the `yield` runs once when the server starts up.
    Code AFTER the `yield` runs once when the server shuts down.


    """
    # --- Startup ---
    logger.info("=" * 60)
    logger.info("NEW RUN STARTED — Production RAG API")
    logger.info("=" * 60)

    yield  # the app runs while paused here

    # --- Shutdown ---
    logger.info("Production RAG API shutting down...")


# Create the FastAPI application instance
app = FastAPI(
    title="Production RAG API",
    description="Backend API for a multi-document RAG chatbot",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routes from other files ("routers") into the main app.
# This keeps main.py clean — it doesn't need to know the details of
# each endpoint, just which routers exist.
app.include_router(health.router, prefix="/health", tags=["Health"])


@app.get("/")
def root():
    """A simple root endpoint just to confirm the server is alive."""
    logger.info("Root endpoint called")
    return {"message": "Production RAG API is running"}