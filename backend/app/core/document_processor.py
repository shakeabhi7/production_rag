import uuid
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

def process_pdf(file_path:Path,original_filename:str):
    """
    Loads a PDF, splits it into chunks, and tags every chunk with:
        - document_id: a unique UUID for this specific upload (never reused,
        even if another file with the same name is uploaded later)
        - filename: the original filename, kept for display purpose only
    Return (document_id,chunks) so the caller can both store the chunks and report
    back the new documnet's ID.
    """

    # Generate a unique ID for this document BEFORE Processing . so we can 
    # tag every chunk with it as we go.

    document_id = str(uuid.uuid4())

    logger.info(f"Loading PDF '{original_filename}' (documnet_id = {document_id})")
    loader = PyMuPDFLoader(str(file_path))
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from '{original_filename}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Tag every chunk's metadata with document_id + filename
    # This is what lets up later filter/deletes by a specific document.
    for chunk in chunks:
        chunk.metadata['document_id'] = document_id
        chunk.metadata["filename"] = original_filename

        # pyMuPPDFLoader already adds a "page" key to metadata

    logger.info(f"Splits '{original_filename} into {len(chunks)} chunks")

    return document_id, chunks