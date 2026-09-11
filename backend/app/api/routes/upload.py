from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.core.document_processor import process_pdf
from app.core.vectorstore import add_documents_batched
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

#Make sure the upload directory exists bedore we try to save file sinto it.
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF, processes it (load->chunk->embed), and stores it
    in the vectorstore, Returns the new document's unique ID
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Only PDF files are supported")

    #save the uploaded file to disk first, since our PDF loader needss a
    # file path (it can't read directly from in=memory upload stream)

    saved_path = UPLOAD_DIR / file.filename
    try:
        contents = await file.read()
        with open(saved_path,"wb") as f:
            f.write(contents)
        logger.info(f"Saved Upload file to '{saved_path}'")
    except Exception as e:
        logger.error(f"Failed to dave uploaded file '{file.filename}': {e}")
        raise HTTPException(status_code=500,detail="Failed to save uploaded file")
    
    # Process: Load -> split -> ta with a unique document_id
    try:
        document_id, chunks = process_pdf(saved_path,file.filename)
    except Exception as e:
        logger.error(f"Failed to process '{file.filename}' : {e}")
        raise HTTPException(status_code=500, detail = f"Failed to process PDF: {e}")

    # Embed + store in ChromaDB
    try:
        added_count = add_documents_batched(chunks)
        logger.info(
            f"Added {added_count} chunks to vectorstore for"
            f"document_id = {document_id} ('{file.filename}')"
        )

    except Exception as e:
        logger.error(f"Failed to embed/store '{file.filename}' : {e}")
        raise HTTPException(status_code=500,detail=f"Failed to embed document: {e}")

    return {
        "document_id" : document_id,
        "filename" : file.filename,
        "chunks_created" : len(chunks),
        "status" : "success"
    }