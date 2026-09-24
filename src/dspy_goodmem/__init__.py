"""GoodMem integration for DSPy: a retriever and agent tools."""

from dspy_goodmem import filters
from dspy_goodmem._filters import GoodMemFilterError
from dspy_goodmem._results import (
    INFORMATIONAL_CODES,
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalStatus,
)
from dspy_goodmem._uploads import GoodMemUploadError
from dspy_goodmem.client import GoodMemClient, GoodMemError
from dspy_goodmem.retriever import GoodMemRM
from dspy_goodmem.tools import make_goodmem_tools

__version__ = "0.2.0"

__all__ = [
    "GoodMemRM",
    "GoodMemClient",
    "GoodMemError",
    "GoodMemFilterError",
    "GoodMemUploadError",
    "RetrievalHit",
    "RetrievalOutcome",
    "RetrievalStatus",
    "INFORMATIONAL_CODES",
    "MALFORMED_STREAM_CODE",
    "UNKNOWN_CODE",
    "make_goodmem_tools",
    "filters",
    "__version__",
]
