import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from app.core.request_context import get_request_id

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class RequestIDFilter(logging.Filter):
    """
    A logging Filter runs on EVERY log line before it's printed/saved.
    This one injects the current request's ID into each log record,
    so every existing logger.info(...) call in the whole app AUTOMATICALLY
    includes the request ID — without us having to edit health.py, chat.py,
    upload.py, etc. to pass it in manually.
    """
    def filter(self, record):
        record.request_id = get_request_id()
        return True


file_handler = TimedRotatingFileHandler(
    filename=LOG_DIR / "app.log",
    when="midnight",
    backupCount=14,
    encoding="utf-8",
)

file_handler.suffix = "%Y-%m-%d"

# Added [%(request_id)s] to the format — this is the only change here
# (Hinglish: format string mein bas request_id add kiya hai)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | [%(request_id)s] | %(name)s | %(message)s"
)
file_handler.setFormatter(formatter)
file_handler.addFilter(RequestIDFilter())

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.addFilter(RequestIDFilter())

logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
# Chroma's telemetry client logs an ERROR on every call because of a
# version mismatch inside its own dependencies. It doesn't affect our app,
# so we raise only that logger's threshold to hide the noise.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)