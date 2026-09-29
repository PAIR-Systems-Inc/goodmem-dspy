"""GoodMem integration for DSPy: a retriever and agent tools."""

from goodmem_dspy import filters
from goodmem_dspy._filters import GoodMemFilterError
from goodmem_dspy._ids import GoodMemIdError
from goodmem_dspy._results import (
    INFORMATIONAL_CODES,
    MALFORMED_STREAM_CODE,
    RERANKING_FAILED_CODE,
    UNKNOWN_CODE,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalStatus,
)
from goodmem_dspy._uploads import GoodMemUploadError
from goodmem_dspy.client import GoodMemClient, GoodMemError
from goodmem_dspy.retriever import GoodMemPassages, GoodMemRM
from goodmem_dspy.tools import make_goodmem_tools

__version__ = "0.3.0"

__all__ = [
    "GoodMemRM",
    "GoodMemPassages",
    "GoodMemClient",
    "GoodMemError",
    "GoodMemFilterError",
    "GoodMemIdError",
    "GoodMemUploadError",
    "RetrievalHit",
    "RetrievalOutcome",
    "RetrievalStatus",
    "INFORMATIONAL_CODES",
    "MALFORMED_STREAM_CODE",
    "RERANKING_FAILED_CODE",
    "UNKNOWN_CODE",
    "make_goodmem_tools",
    "filters",
    "__version__",
]
