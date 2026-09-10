from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Return a configured Gemini embeddings object.

    
    This is kept as a small standalone function (rather than creating the
    embeddings object inline wherever it's needed) so every part of the app
    — vectorstore setup, ingestion, retrieval — uses the exact same
    embedding configuration. If we ever need to switch embedding models,
    we only change it here.
    """
    return GoogleGenerativeAIEmbeddings(
        model = settings.embeddings_model,
        google_api_key=settings.google_api_key,
    )