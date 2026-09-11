import os
import time

# Disable Chroma's anonymous telemetry via environment variable. This is
# the most reliable way to disable it — passing client_settings directly
# to Chroma() doesn't always propagate correctly depending on the
# langchain-chroma version. Must be set BEFORE chromadb is imported/used.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma

from app.core.config import settings
from app.core.embeddings import get_embeddings
from app.core.logger import get_logger

logger = get_logger(__name__)

# A single shared Chroma client/collection for the whole app.
# Unlike FAISS (which we saved/loaded manually as files), Chroma manages
# its own on-disk persistence automatically — we just point it at a
# directory (settings.chroma_persist_dir) and it handles reading/writing.
_vectorstore = None

# --- Batching settings (same reasoning as the Simple RAG project) ---
# Gemini's free tier embedding quota is 100 requests/minute. We stay
# comfortably under that per batch, and wait between batches so we don't
# get RESOURCE_EXHAUSTED (429) errors on larger documents.
BATCH_SIZE = 80
DELAY_BETWEEN_BATCHES = 60  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds


def get_vectorstore() -> Chroma:
    """
    Returns the shared Chroma vectorstore instance, creating it on first call.

    Chroma automatically persists to disk at `settings.chroma_persist_dir`
    (see config.py) — there's no separate "save" step like we had with
    FAISS. Every `add_documents()` call is saved immediately.
    """
    global _vectorstore

    if _vectorstore is None:
        logger.info(f"Initializing ChromaDB at '{settings.chroma_persist_dir}'")
        _vectorstore = Chroma(
            collection_name="documents",
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )

    return _vectorstore


def add_documents_batched(chunks) -> int:
    """
    Embeds and stores chunks in the vectorstore in batches, to avoid
    hitting Gemini's free-tier rate limit (100 requests/minute) on larger
    documents. Retries a batch with a delay if it fails transiently.

    Returns the number of chunks successfully added.
    """
    vectorstore = get_vectorstore()
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Embedding {len(chunks)} chunks in {total_batches} batches of {BATCH_SIZE}")

    added = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = chunks[i:i + BATCH_SIZE]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vectorstore.add_documents(batch)
                added += len(batch)
                logger.info(f"Batch {batch_num}/{total_batches} embedded ({len(batch)} chunks)")
                break
            except Exception as e:
                logger.warning(f"Batch {batch_num} failed on attempt {attempt}: {e}")
                if attempt < MAX_RETRIES:
                    logger.info(f"Waiting {RETRY_DELAY}s before retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Batch {batch_num} failed after {MAX_RETRIES} attempts.")
                    raise

        if batch_num < total_batches:
            logger.info(f"Waiting {DELAY_BETWEEN_BATCHES}s before next batch (rate limit cooldown)...")
            time.sleep(DELAY_BETWEEN_BATCHES)

    logger.info(f"Finished embedding: {added}/{len(chunks)} chunks added")
    return added