import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


file_handler = TimedRotatingFileHandler(
    filename=LOG_DIR / "app.log",
    when = "midnight",
    backupCount=15,
    encoding="utf-8"
)

file_handler.suffix = "%Y-%m-%d"

# format explanation:
#   %(asctime)s   -> timestamp, e.g. 2026-09-09 14:32:10
#   %(levelname)s -> INFO / WARNING / ERROR / etc.
#   %(name)s      -> which module logged this (e.g. app.api.routes.health)
#   %(message)s   -> the actual log message

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler() # prints to the terminal
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler,file_handler]
)

def get_logger(name : str,) -> logging.Logger:
    """
    Returns a logger scoped to the given module name.
    """
    return logging.getLogger(name)
