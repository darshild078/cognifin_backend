import os
import sys
import logging
import warnings
from typing import Optional
from contextvars import ContextVar

# Suppress deprecation and future warnings for clean output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment flags to prevent noisy third-party outputs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Thread-safe request ID context variable
request_id_ctx_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class SingleLineFormatter(logging.Formatter):
    """Custom structured single-line log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = request_id_ctx_var.get() or "-"
        record.request_id = req_id
        return super().format(record)


def setup_logging(level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    formatter = SingleLineFormatter(
        fmt="%(asctime)s [%(levelname)s] [req_id=%(request_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Silence noisy third-party libraries
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
        "pymongo",
        "multipart",
        "asyncio",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("cognifin")
