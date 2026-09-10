from langchain_chroma import Chroma

from app.core.config import settings
from app.core.embeddings import get_embeddings
from app.core.logger import get_logger

logger = get_logger(__name__)

# A Single shared Chroma client/collection for the whole app.
# unlike FAISS (which we saved/loaded manually as files), chroma manages
# its own on-disk persistence automatically - we just point it at a
# directory (settings.chroma_presist_dir) and it handles reading/writing.
_vectorstore = None

def get_vectorstore() -> Chroma:
    """
    Returns the shared Chroma vectorstore instance, creating it on first call.

    Chroma automatically persists to disk at 'settings.chroma_persist_dir'
    (see config.py) - there's no separate "save" step like we had with 
    FAISS. Every 'add_document()' call is savec immediately.
    """
    global _vectorstore

    if _vectorstore is None:
        logger.info(f"Initializing ChromaDB as '{settings.chroma_persist_dir}'")
        _vectorstore = Chroma(
            collection_name="documents",
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir
        )

    return _vectorstore