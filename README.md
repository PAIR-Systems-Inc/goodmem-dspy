# dspy-goodmem

[![PyPI version](https://img.shields.io/pypi/v/dspy-goodmem.svg)](https://pypi.org/project/dspy-goodmem/)
[![Python versions](https://img.shields.io/pypi/pyversions/dspy-goodmem.svg)](https://pypi.org/project/dspy-goodmem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[GoodMem](https://goodmem.ai) integration for [DSPy](https://dspy.ai) — a self-hosted memory backend for RAG pipelines and agents.

Ships a DSPy retriever (`GoodMemRM`), a raw HTTP client (`GoodMemClient`), and a tool factory (`make_goodmem_tools`) that exposes GoodMem's full space/memory lifecycle to `dspy.ReAct` agents — not just retrieval.

## Installation

```bash
pip install dspy-goodmem
```

You also need a running GoodMem server. See [docs.goodmem.ai](https://docs.goodmem.ai) for setup.

## Quick start

```python
import dspy
from dspy_goodmem import GoodMemRM

# Configure your LM (any LiteLLM-supported provider works)
dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

# Create a retriever backed by GoodMem
rm = GoodMemRM(
    space_ids=["<your-space-uuid>"],
    api_key="gm_...",
    base_url="https://localhost:8080",
    k=3,
)

# Build a simple RAG module
class RAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        passages = self.retriever(question)
        context = "\n\n".join(p["long_text"] for p in passages)
        return self.respond(context=context, question=question)

rag = RAG(retriever=rm)
print(rag(question="What are the Series B terms?").response)
```

## What's included

| Export | Role |
|---|---|
| `GoodMemRM` | `dspy.Retrieve` subclass — returns `dotdict({"long_text": ...})` passages for any DSPy pipeline |
| `GoodMemClient` | Low-level HTTP wrapper around all 11 GoodMem REST operations |
| `make_goodmem_tools` | Factory that produces 11 typed callables for `dspy.Tool` / `dspy.ReAct` |

## Agent memory lifecycle

Unlike retriever-only integrations, `make_goodmem_tools` lets a `dspy.ReAct` agent **manage its own memory** — create spaces, store new memories, retrieve, update, and delete — without a human in the loop:

```python
import dspy
from dspy_goodmem import GoodMemClient, make_goodmem_tools

client = GoodMemClient(api_key="gm_...", base_url="https://localhost:8080")
tools = [dspy.Tool(fn) for fn in make_goodmem_tools(client)]

agent = dspy.ReAct("task -> result", tools=tools)
agent(task="Remember that the user prefers Python over Java, then recall my language preferences.")
```

## GoodMem features you gain

- **Bring your own embedding model** — OpenAI, Voyage AI, Cohere, vLLM, TEI, Llama.cpp, including fully local/offline models
- **Hybrid search** — combine dense and sparse embedders (e.g. MiniLM + SPLADE) in a single space with configurable weights
- **Configurable chunking** — chunk size, overlap, separators, and mode set per space and handled on the server
- **File ingestion** — upload PDFs, DOCX, images, and other formats without manual text extraction
- **Metadata filtering** — SQL-style filters with JSONPath extraction, date ranges, regex, and array membership
- **Reranking** — pluggable reranker models re-score results after retrieval
- **Auto-summary** — an LLM generates a consolidated answer from retrieved chunks at query time
- **Deep Research mode** — multiple iterative search rounds with query refinement for open-ended topics
- **Self-hosted** — runs entirely on your infrastructure, no external API calls or quotas

## Full examples

Two end-to-end examples cover the main usage patterns:

- [`examples/rag_pipeline_example.py`](examples/rag_pipeline_example.py) — Classic RAG pipeline using `GoodMemRM` as a retriever with `ChainOfThought` and `SemanticF1` evaluation.
- [`examples/react_agent_example.py`](examples/react_agent_example.py) — Agent-driven memory with `dspy.ReAct` and `make_goodmem_tools`, covering multi-turn conversation, cross-agent persistence, metadata filtering, and trajectory inspection.

Both load a `.env` file from the repo root if [python-dotenv](https://pypi.org/project/python-dotenv/) is installed (`pip install dspy-goodmem[examples]`), otherwise they read environment variables directly.

```bash
export OPENAI_API_KEY="sk-..."
export GOODMEM_API_KEY="gm_..."
export GOODMEM_BASE_URL="https://localhost:8080"

python examples/rag_pipeline_example.py
# or
python examples/react_agent_example.py
```

## Development

```bash
pip install -e .[dev]
pytest tests/ -v
```

63 mocked unit tests cover the client, retriever, and tool factory — no live server required.

## Related

- [DSPy documentation](https://dspy.ai)
- [GoodMem documentation](https://docs.goodmem.ai)

## License

MIT — see [LICENSE](LICENSE).
