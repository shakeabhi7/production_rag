from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# In-memory store: session_id -> that session's chat history.
# This is intentionally simple for now — it lives in RAM, so history is
# lost if the server restarts. In a later session we'll swap this for a
# proper database (so history survives restarts and works across multiple
# server instances). For learning/local use, this is perfectly fine.

_store: dict[str,BaseChatMessageHistory] = {}

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    """
    Returns the chat history for the given session_id, creating a new
    one if this session hasn't been seen before.

    LangChain's RunnableWithMessageHistory calls this function automatically
    - we don't call it directly. It's how the chain knows "Whose conversation" it's continuing.
    """
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]