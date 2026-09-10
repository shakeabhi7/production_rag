from fastapi import APIRouter, HTTPException
from app.core.vectorstore import get_vectorstore
from app.core.logger import get_logger

logger = get_logger(__name__)
router =APIRouter()

@router.get("/")
def list_documents():
    """
    List all unique documents currently stored in the vectorstore
    (grouped by documnet_id, since each documents has many chunks).
    """

    vectorstore = get_vectorstore()

    # chroma's get() with no filter returns enverything in the collection,
    # including each chunks's metadata (where document_id + filename live).
    all_data = vectorstore.get(include=["metadatas"])

    # multiple chunks share the same document_id - collapse them into one
    # entry per document, and count how many chunks each document has.

    documents = {}
    for metadata in all_data["metadatas"]:
        doc_id = metadata.get("document_id")
        filename = metadata.get("filename")
        if doc_id not in documents:
            documents[doc_id] = {"document_id":doc_id,"filename":filename,"chunk_count":0}
            documents[doc_id]["chunk_count"] +=1
    return {"documents": list(documents.values()),"total_documents":len(documents)}

@router.delete("/{documnent_id}")
def delete_document(documnet_id:str):
    """
    Deletes all chunks belonging to a specific document_id from the
    vectorstore. This removes the documnet from future retrieval without 
    touching any other documents.
    """
    vectorstore = get_vectorstore()

    # check the document actually exists before attempting deletion, so we
    # can return a clear 404 instead of silently doing nothing
    existing  =  vectorstore.get(where={"document_id":documnet_id},include=["metadatas"])
    if not existing["ids"]:
        raise HTTPException(status_code=404,detail=f"No document found with id '{documnet_id}")

    chunk_count = len(existing["ids"])
    vectorstore.delete(ids=existing["ids"])

    logger.info(f"Deleted document_id = {documnet_id} ({chunk_count} chunks removed)")

    return {"document_id": documnet_id,"chunks_deleted":chunk_count,"status":"deleted"}

