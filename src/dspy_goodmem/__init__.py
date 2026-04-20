"""GoodMem integration for DSPy.

Provides a retriever, HTTP client, and tool factory for using GoodMem
(https://goodmem.ai) as a memory backend in DSPy pipelines and agents.

Main exports:
    GoodMemRM: DSPy retriever backed by GoodMem semantic search
    GoodMemClient: HTTP client wrapping the GoodMem REST API
    make_goodmem_tools: Factory producing callables for dspy.ReAct agents
"""

from dspy_goodmem.client import GoodMemClient
from dspy_goodmem.retriever import GoodMemRM
from dspy_goodmem.tools import make_goodmem_tools

__version__ = "0.1.0"
__all__ = ["GoodMemClient", "GoodMemRM", "make_goodmem_tools"]
