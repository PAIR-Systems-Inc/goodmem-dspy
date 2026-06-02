# goodmem-dspy

[![PyPI](https://img.shields.io/pypi/v/dspy-goodmem.svg)](https://pypi.org/project/dspy-goodmem/)
[![Python](https://img.shields.io/pypi/pyversions/dspy-goodmem.svg)](https://pypi.org/project/dspy-goodmem/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[GoodMem](https://goodmem.ai) integration for [DSPy](https://dspy.ai).

GoodMem is a self-hosted RAG system which handles the full retrieval pipeline: ingestion, chunking, embedding, storage, hybrid search, reranking, and summarization. This package wraps it for DSPy so you can:

- Plug `GoodMemRM` into any DSPy pipeline through the standard `dspy.Retrieve` interface.
- Hand GoodMem's full memory lifecycle to a `dspy.ReAct` agent as callable tools.
- Use `GoodMemClient` directly when you want control over the REST API.

## Install

```bash
pip install dspy-goodmem
```

A running GoodMem server is required. See the [Quick Start](https://goodmem.ai/quick-start) for deployment instructions.

## Retriever usage

```python
import dspy
from dspy_goodmem import GoodMemRM

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

rm = GoodMemRM(
    space_ids=["<your-space-uuid>"],
    api_key="gm_...",
    base_url="https://localhost:8080",
    k=3,
    verify_ssl=False,  # localhost self-signed cert; remove for a server with a valid TLS cert
)

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
print(rag(question="Summarize what's in the knowledge base.").response)
```

## Agent usage

`make_goodmem_tools` returns 11 plain callables covering every GoodMem operation: full CRUD for spaces, create/list/get/delete for memories, plus semantic retrieval and embedder discovery. Wrap them in `dspy.Tool` and a `dspy.ReAct` agent can manage its own memory end to end instead of only reading from it.

```python
import dspy
from dspy_goodmem import GoodMemClient, make_goodmem_tools

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

client = GoodMemClient(
    api_key="gm_...",
    base_url="https://localhost:8080",
    verify_ssl=False,  # localhost self-signed cert; remove for a server with a valid TLS cert
)
tools = [dspy.Tool(fn) for fn in make_goodmem_tools(client)]

agent = dspy.ReAct("task -> result", tools=tools)
agent(task="Remember that the user prefers Python over Java, then recall their language preferences.")
```

## What's exported

| Export | Purpose |
|---|---|
| `GoodMemRM` | `dspy.Retrieve` subclass. Returns `dotdict({"long_text": ...})` passages. |
| `GoodMemClient` | HTTP wrapper around the GoodMem REST API. |
| `make_goodmem_tools` | Factory that produces typed callables for `dspy.Tool` and `dspy.ReAct`. |

## Examples

Two end-to-end scripts live in `examples/`:

- [`rag_pipeline_example.py`](examples/rag_pipeline_example.py) runs `GoodMemRM` through `ChainOfThought` and scores the pipeline with `SemanticF1`.
- [`react_agent_example.py`](examples/react_agent_example.py) exercises the ReAct tools across four scenarios: multi-turn conversation, cross-agent persistence, metadata-tagged filtering, and trajectory inspection.

Both scripts load a `.env` file at the repo root if `python-dotenv` is installed (`pip install dspy-goodmem[examples]`). Otherwise they read environment variables directly.

```bash
export OPENAI_API_KEY="sk-..."
export GOODMEM_API_KEY="gm_..."
export GOODMEM_BASE_URL="https://localhost:8080"

python examples/rag_pipeline_example.py
python examples/react_agent_example.py
```

## Why GoodMem

GoodMem does the heavy lifting server side, so you don't ship an embedding pipeline with your DSPy app:

- Supports OpenAI, Voyage, Cohere, vLLM, TEI, and Llama.cpp as embedders, including fully local models.
- Hybrid search combining dense and sparse embedders, with configurable weights per space.
- Per-space chunking config: size, overlap, separators.
- Native ingestion for plain text, PDFs, Word documents, images, spreadsheets, and other formats.
- Metadata filters with JSONPath extraction, regex, and date ranges.
- Reranking and auto-summary pipelines configurable per request.
- Deep Research mode runs multiple iterative search rounds with query refinement for complex and open-ended topics.

Everything runs on your own infrastructure, so documents and queries never leave your network.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

63 mocked unit tests cover the client, retriever, and tool factory. No live server required.

## Links

- [DSPy](https://dspy.ai)
- [GoodMem](https://goodmem.ai) ([docs](https://docs.goodmem.ai))
- [Issues](https://github.com/PAIR-Systems-Inc/dspy-goodmem/issues)

## License

MIT. See [LICENSE](LICENSE).
