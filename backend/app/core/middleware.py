import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.request_context import set_request_id
from app.core.logger import get_logger

logger = get_logger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Runs for Every request, automatically:
    1. Generates a request ID
    2. Logs when the request starts
    3. Times how long it takes
    4. Logs when it finishes, with the status code
    5. Adds the requests ID to the response headers (X-Request-ID) 
    """

    async def dispatch(self, request:Request, call_next):
        request_id = set_request_id()
        start_time = time.time()

        logger.info(f"Request started | {request.method} {request.url.path}")

        try:
            response = await call_next(request)
        except Exception:
            # Without this, a crashed request never gets a "completed" log line.
    
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(
                f"Request failed | {request.method} {request.url.path} | "
                f"status=500 | duration={duration_ms}ms"
            )
            raise

        duration_ms = round((time.time() - start_time) * 1000,2)
        logger.info(
            f"Request completed | {request.method} {request.url.path} |"
            f"status={response.status_code} | duration={duration_ms}ms"
        )

        response.headers["X-Request-ID"] = request_id
        return response
        