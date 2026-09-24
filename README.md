# dspy-goodmem

[GoodMem](https://docs.goodmem.ai) memory for [DSPy](https://github.com/stanfordnlp/dspy):
a retriever and a set of agent tools. Documents are chunked, embedded and
searched server-side; this package wraps the official `goodmem` Python SDK.

**Version 0.2.0.** Verified against GoodMem server **v1.0.320**.

> **Upgrading from 0.1.1.** 0.1.1 talked to GoodMem over hand-written HTTP. A
> retrieval that *failed* — a space whose embedder was unavailable, say —
> returned `success: true` with zero results and no indication anything had
> gone wrong, indistinguishable from an empty index. See
> [Changes in 0.2.0](#changes-in-020).

## Install

```bash
pip install dspy-goodmem
export GOODMEM_API_KEY="gm_your_key_here"
export GOODMEM_BASE_URL="https://your-goodmem-server"
```

## Retrieve

```python
import dspy
from dspy_goodmem import GoodMemRM

rm = GoodMemRM(space_ids=["<space-uuid>"], k=3)
dspy.settings.configure(rm=rm)

passages = rm("What is the main finding?").passages
```

Each passage is a `dotdict` carrying `long_text` — what DSPy consumes —
alongside the data 0.1.1 discarded:

```python
{
  "long_text": "...",         # the chunk text
  "score": 0.64,              # higher is better
  "raw_score": -0.64,         # exactly what the server sent
  "score_kind": "vector",     # or "reranker" -- not the same scale
  "chunk_id": "...", "memory_id": "...", "space_id": "...",
  "metadata": {...},          # the memory's metadata, joined by UUID
  "goodmem_partial": False,   # True when the server reported a problem
}
```

### When retrieval goes wrong

A degraded retrieval still returns whatever passages arrived, each flagged
with `goodmem_partial`. A `dspy.Prediction` has no slot for a flag on an
*empty* result, so a degraded retrieval that returns nothing raises a
`UserWarning` and logs at WARNING with the server's own reason — rather than
looking like a clean miss:

```
UserWarning: GoodMem reported a problem during retrieval and returned no
passages -- this is not an empty index: EMBEDDER_FAILED: Embedding failed
```

### Scores

GoodMem produces two kinds of score and they are not comparable. **Vector**
scores are negative distances, so `score` is the flipped value with
`raw_score` kept beside it. **Reranker** scores are already higher-is-better,
on a **provider-dependent** scale — measured live on the same five documents,
Voyage `rerank-2.5` returned `0.27..0.93` and Jina `jina-reranker-v3` returned
`-0.14..0.43`.

So there is **no default threshold**; `min_score` applies only when
`reranker_id` is set, and warns naming the observed range if it removes
everything.

## Agent tools

```python
import dspy
from dspy_goodmem import GoodMemClient, make_goodmem_tools

client = GoodMemClient()
tools = make_goodmem_tools(client, space_ids=["<space-uuid>"])
agent = dspy.ReAct("question -> answer", tools=[dspy.Tool(t) for t in tools])
```

By default the agent gets exactly two tools — `goodmem_search(query, top_k)`
and `goodmem_remember(text)`. The model never chooses a space. 0.1.1 handed it
eleven, including `delete_space` and a `public_read` argument the server
rejects.

| Argument | Adds |
| --- | --- |
| `allow_upload=True` | `goodmem_upload_file`, confined to the client's `upload_dir` |
| `allow_admin=True` | space/embedder management and `get_memory` |
| `allow_delete=True` | `delete_memory`, `delete_space` |
| `allow_write=False` | removes `goodmem_remember` |

## Metadata filters

Filters are expressions evaluated server-side, not SQL. Build them with the
`filters` helper rather than by string interpolation:

```python
from dspy_goodmem import GoodMemRM, filters

rm = GoodMemRM(space_ids=["..."], metadata_filter={"tenant": "acme", "active": True})

expression = filters.all_of(
    filters.equals("tenant", "acme"),
    filters.compare("year", ">=", 2026),
)
```

The helper applies the escaping the server accepts (`'` → `\'`, `\` → `\\`;
SQL-style `''` doubling is rejected with HTTP 400), refuses control
characters, restricts field names, and casts each value to the type GoodMem
stored — a boolean compared as `TEXT` is accepted with HTTP 200 and matches
nothing.

## Uploads

Uploads are **off** unless the client is given an `upload_dir`. When set,
every path is resolved (symlinks included) and refused if it lands outside
that directory, so a model-supplied path cannot read arbitrary host files.

```python
client = GoodMemClient(upload_dir="/srv/agent-uploads")
```

## Changes in 0.2.0

Reproduced against the published 0.1.1 wheel, live against GoodMem v1.0.320.

| Was | Now |
| --- | --- |
| Hand-written `requests` client | Official `goodmem` SDK |
| A space with a failing embedder returned `success: true, totalResults: 0` — the server's `EMBEDDER_FAILED` was dropped, so a failed search looked like an empty index | `partial` + `statuses`, a `UserWarning` and a WARNING log carrying the server's reason |
| The retriever handed DSPy `dotdict({"long_text": ...})` and nothing else — no score, no ids, no metadata | Every passage carries score, `score_kind`, ids and metadata |
| `public_read` was a model-facing tool argument; the server answers `400 Unrecognized field "publicRead"` | Not offered anywhere |
| Eleven agent tools including `delete_space` | `goodmem_search` + `goodmem_remember`; the rest opt-in |
| `file_path` was an unrestricted tool argument; it read `/etc/hostname` and uploaded it | Confined to `upload_dir`; `..` and symlink escapes refused |
| Empty search took **10.88 s** — `wait_for_indexing` on by default | **0.33 s**; the read path never polls |
| **0 of 13** HTTP calls carried a timeout | On the client, configurable |
| Chunks and memories were two arrays joined by position | Joined by UUID, de-duplicated by chunk id |
| Raw negative scores | `score` / `raw_score` / `score_kind` |
| Reusing a space name silently accepted a different embedder and reported the one you asked for | Reuse requires a match; a mismatch names both |
| `list_spaces` returned the first page; `nextToken` appeared nowhere | Paginated, bounded by `max_list_items` |
| No metadata filtering | `filters`, escaped and type-correct |
| Content type guessed from the **host's** `/etc/mime.types`, so the same file could upload as a different type on another machine — and the package's own test for it failed on this one | The SDK decides; content is decoded by the type the server reports |
| 62 tests patching `requests`, green against every defect; no CI | 52 offline + 19 live; CI on 3.10–3.13 |

## Tests

| Suite | Count | Needs |
| --- | --- | --- |
| `tests/test_dspy_goodmem.py` | 52 | nothing — the real SDK over a mock transport, fed NDJSON captured from a live server |
| `tests/test_dspy_goodmem_live.py` | 19 | `GOODMEM_API_KEY` + `GOODMEM_BASE_URL`; skips entirely without them |

```bash
pip install -e . pytest httpx "ruff==0.7.4" mypy

pytest tests/test_dspy_goodmem.py

GOODMEM_API_KEY=... GOODMEM_BASE_URL=... \
  GOODMEM_TEST_EMBEDDER_ID=... \
  pytest tests/test_dspy_goodmem_live.py

# what CI runs
ruff check src tests && ruff format --check src tests && mypy src/dspy_goodmem
```

The live suite creates one space per run and asserts, against a fresh server
listing, that it is gone afterwards.

## A note on TLS

`verify_ssl` exists for self-signed development servers and defaults to on.
0.1.1's README and the retriever's own docstring both showed it turned off, so
that is what people copied; no example here does, and CI fails if one appears.

## License

MIT.
