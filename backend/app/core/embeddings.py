import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# --- Retry settings for single-query embedding ---
# It's transient — retrying
# almost always succeeds on the 2nd or 3rd attempt. This wrapper adds
# automatic retries specifically around embed_query (used during chat/
# retrieval), separate from the batch upload retry logic in vectorstore.py.
QUERY_RETRY_ATTEMPTS = 3
QUERY_RETRY_DELAY = 2  # seconds


class RetryingGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    A thin wrapper around GoogleGenerativeAIEmbeddings that retries
    embed_query() a few times on failure, with a short delay between
    attempts. embed_documents() (used during bulk upload) is left as-is
    since that already has its own batching + retry logic one layer up.
    """

    def embed_query(self, text: str) -> list[float]:
        last_error = None
        for attempt in range(1, QUERY_RETRY_ATTEMPTS + 1):
            try:
                return super().embed_query(text)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"embed_query failed on attempt {attempt}/{QUERY_RETRY_ATTEMPTS}: {e}"
                )
                if attempt < QUERY_RETRY_ATTEMPTS:
                    time.sleep(QUERY_RETRY_DELAY)
        # All attempts failed — raise the last error so the caller still
        # sees a clear failure instead of this being silently swallowed.
        raise last_error


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Returns a configured Gemini embeddings object.

    This is kept as a small standalone function (rather than creating the
    embeddings object inline wherever it's needed) so every part of the app
    — vectorstore setup, ingestion, retrieval — uses the exact same
    embedding configuration. If we ever need to switch embedding models,
    we only change it here.
    """
    return RetryingGeminiEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )