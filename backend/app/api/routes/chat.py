from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.chains import build_conversational_chain
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """
    Request body for a chat message.

    - question: the user's message
    - session_id: identifies WHICH conversation this belongs to, so
      follow-up questions get the right history. The frontend should
      generate/reuse a consistent session_id per conversation (e.g. a
      random ID created when a chat window is opened).
    - document_id: optional. If provided, retrieval is scoped to only that
      document. If omitted, retrieval searches across all documents.
    """
    question: str
    session_id: str = "default"
    document_id: Optional[str] = None


@router.post("/")
def chat(request: ChatRequest):
    # Sanitize input — an empty/whitespace-only question wastes an API
    # call and produces a confusing downstream error, so reject it early
    # with a clear message instead.
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(
        f"Chat request | session={request.session_id} | "
        f"document_id={request.document_id} | question='{question}'"
    )

    chain = build_conversational_chain(request.document_id)

    try:
        result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": request.session_id}},
        )
    except Exception as e:
        # Gemini's embedding API is known to be intermittently unstable
        # right now (see core/embeddings.py) — even with retries, a request
        # can still fail. Rather than returning a raw 500 with a scary
        # traceback, we return a clear, actionable error to the caller.
        logger.error(f"Chat request failed after retries: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable (this is a "
                "known intermittent issue on Google's side). Please try "
                "again in a few seconds."
            ),
        )

    # `result["context"]` contains the retrieved Document chunks used for
    # this answer — we pull out their filenames to show as sources.
    sources = sorted(set(
        doc.metadata.get("filename", "unknown")
        for doc in result.get("context", [])
    ))

    return {
        "answer": result["answer"],
        "sources": sources,
        "session_id": request.session_id,
    }