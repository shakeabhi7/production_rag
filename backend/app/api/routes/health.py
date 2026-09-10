from fastapi import APIRouter
from app.core.logger import get_logger
from app.core.config import settings

# An APIRouter groups related endpoints together. Instead of putting every
# endpoint directly in main.py, we split them by feature (health, upload, chat..)
# and "include" each router into main page

router = APIRouter()
logger = get_logger(__name__)

@router.get("/")
def health_check():
    """
    Basic health check = confirms the server is running and that
    settings (including the required GOOGLE_API_KEY) loaded correctly
    """
    logger.info("Health check endpoint called")
    return {
        "status" : "OK",
        "llm_model" : settings.llm_model,
        "embedding_model" : settings.embeddings_model,
        
    }