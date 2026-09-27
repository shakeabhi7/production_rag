import contextvars
import uuid

_request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="no-request-id"
)

def set_request_id() -> str:
    """
    Generate a new request ID and stores it for the CURRENT request only.
    Called once per request, at the very start(in middleware).
    """

    request_id = str(uuid.uuid4())[:8]
    _request_id_ctx_var.set(request_id)
    return request_id

def get_request_id() -> str:
    """Returns the current request'sID, whereever this is called from."""
    return _request_id_ctx_var.get()