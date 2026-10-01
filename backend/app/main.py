from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


from app.api.routes import health,upload,documents,chat
from app.core.logger import get_logger

from app.core.middleware import LoggingMiddleware
from app.core.request_context import get_request_id

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
app.add_middleware(LoggingMiddleware)

# Register routes from other files ("routers") into the main app.
# This keeps main.py clean — it doesn't need to know the details of
# each endpoint, just which routers exist.
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(documents.router,prefix="/documents",tags=["Document"])
app.include_router(chat.router, prefix="/chat",tags=["Chat"])

@app.get("/")
def root():
    """A simple root endpoint just to confirm the server is alive."""
    logger.info("Root endpoint called")
    return {"message": "Production RAG API is running"}

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request:Request,exc:StarletteHTTPException):
    """
    Standardize ALL HTTPException responses into one consistent shape,
    instead of FastAPI's default
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"error":exc.detail,"request_id":get_request_id()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request:Request,exc:Exception):
    """
    Catches ANY unhandled exception, anywhere in the app - the ones we didn't specifically 
    anticipate with try/except. Logs the full traceback for debugging, but shows the user 
    only a clean, generic message(no internal details leaked).
    """
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error":"An unexpected error occurred. Please try again.",
            "request_id": get_request_id()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request data",
            "details": jsonable_encoder(exc.errors()),
            "request_id": get_request_id(),
        },
    )