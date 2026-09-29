# goodmem-dspy

[GoodMem](https://docs.goodmem.ai) memory for [DSPy](https://github.com/stanfordnlp/dspy):
a retriever and a set of agent tools. Documents are chunked, embedded and
searched server-side; this package wraps the official `goodmem` Python SDK.

**Version 0.3.0.** 0.2.0 was verified against GoodMem server **v1.0.320**;
0.3.0 against the monthly end-to-end run (dspy 3.4.0, 17 of 17 checks).

## Install

```bash
pip install goodmem-dspy
export GOODMEM_API_KEY="gm_your_key_here"
export GOODMEM_BASE_URL="https://your-goodmem-server"
```

## Retrieve

```python
import dspy
from goodmem_dspy import GoodMemRM

rm = GoodMemRM(space_ids=["<space-uuid>"], k=3)

# As DSPy's retrieval module: dspy.Retrieve reads dspy.settings.rm.
dspy.settings.configure(rm=rm)
texts = dspy.Retrieve(k=3)("What is the main finding?").passages  # list of str

# Directly, keeping scores, ids and metadata:
passages = rm("What is the main finding?").passages
```

Calling the retriever returns a `GoodMemPassages`: a plain list of passages —
the shape DSPy's own retrievers return and the one `dspy.Retrieve` iterates —
that also answers `.passages`, plus `.partial` and `.statuses` (see below).
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
with `goodmem_partial`, and the result's own `partial` is `True` with the
server's `statuses` beside it. A degraded retrieval that returns *nothing*
comes back as an empty list with `partial` set, and also raises a
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

`score_kind` says which stage the score came from, as the server reported it
— not as configured. When a reranker was requested but could not run, GoodMem
reports `RERANKING_FAILED` (and a `NOT_FOUND` naming the reranker) and still
returns the vector-stage hits. Those are labelled `"vector"`, flipped to
higher-is-better like any vector score, and flagged `goodmem_partial`. A
`NOT_FOUND` naming anything else, such as an LLM, leaves reranker scores as
they are.

So there is **no default threshold**; `min_score` applies only to reranker
scores — with `reranker_id` set and the reranker actually run — and warns
naming the observed range if it removes everything. It never drops a
fallback's vector hits.

## Agent tools

```python
import dspy
from goodmem_dspy import GoodMemClient, make_goodmem_tools

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

Every id argument these tools take is declared a UUID in the schema the model
is shown on DSPy 3.x, and refused by the client if it is not one — see
[Ids](#ids).

## Metadata filters

Filters are expressions evaluated server-side, not SQL. Build them with the
`filters` helper rather than by string interpolation:

```python
from goodmem_dspy import GoodMemRM, filters

rm = GoodMemRM(space_ids=["<space-uuid>"], metadata_filter={"tenant": "acme", "active": True})

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

## Ids

Every id — space, memory, embedder, reranker — must be a canonical UUID
(either case; it is sent lowercase). Anything else raises `GoodMemIdError`, a
`ValueError`, **before any request is made**, because the GoodMem SDK puts ids
into URL paths as they are and `httpx` resolves `..`: on 0.2.0,
`delete_memory("../spaces/<id>")` was sent as `DELETE /v1/spaces/<id>` and
reported success. `GoodMemRM` and `make_goodmem_tools` check their configured
ids when built, and the client checks every id again on every call; that
client check is the guard, on every DSPy version. On DSPy 3.x the opt-in tools
also declare each id argument with the UUID pattern, so the model is told and
`dspy.Tool` refuses a mismatch first. DSPy 2.5 shows the model only the type
name `UUIDStr` and passes arguments through unchecked, so there the client's
`GoodMemIdError` is what refuses.

The id sent is always a plain string made from the characters that were
checked, so a `str` subclass that overrides `lower()` or `__str__` cannot swap
in a different path after the check, and a `uuid.UUID` whose text is not a
UUID is refused. A single `uuid.UUID` is accepted wherever one space id or a
list is.

## Changes in 0.3.0

**Renamed to `goodmem-dspy` (import `goodmem_dspy`)**, the `goodmem-<framework>`
naming used by goodmem-adk and goodmem-semantic-kernel. **Breaking:** update
imports from `dspy_goodmem` to `goodmem_dspy`. 0.2.0 and 0.2.1 were tagged but
never reached PyPI, so 0.3.0 is the first release since 0.1.1.

Two defects were found by the monthly end-to-end run against dspy 3.4.0 and
are measured below with the same script against 0.2.1 and this release: live
against GoodMem at `localhost:8080`, and offline replaying streams captured
from that server on 2026-09-29.

| Was (0.2.1) | Now |
| --- | --- |
| With `GoodMemRM` configured as `dspy.settings.rm` — exactly as this README said — every `dspy.Retrieve(...)` call raised `AttributeError: 'str' object has no attribute 'long_text'`: the retriever returned a `dspy.Prediction`, which iterates as its key names | Returns `GoodMemPassages`, a list of passage dotdicts that also answers `.passages`; `dspy.Retrieve` returns the passage texts, and `rm(q).passages` keeps working |
| A requested reranker that failed (`NOT_FOUND` + `RERANKING_FAILED`) left the vector fallback hits labelled `score_kind: "reranker"` with the raw negative value as `score` (live: `-0.4268`, `-0.1455`), so a higher-is-better reader ranked them backwards | Labelled `"vector"`, `score` `0.4268`, `0.1455`, `raw_score` kept; decided after the whole stream is read |
| `min_score=0.5` on that fallback removed both hits and then warned "this is not an empty index" | Both hits kept, no `min_score` warning; `partial` + statuses as the retrieval status contract says |
| A degraded empty result was indistinguishable from `[]` except through the warning | `result.partial` and `result.statuses` on the returned list |
| 138 offline + 20 live tests | 157 offline + 23 live; 17 of the new offline tests fail on 0.2.1 |

## Changes in 0.2.1

0.2.0 was tagged but never published to PyPI, so 0.2.1 is the first release
of this line to ship. Measured offline with the production client talking to
a local HTTP server that records every request.

| Was (0.2.0) | Now |
| --- | --- |
| `delete_memory("../spaces/<id>")` — a developer call, or the model through `allow_delete=True` — was sent as `DELETE /v1/spaces/<id>` and returned `success: true`; `delete_space`, `get_space`, `update_space`, `get_memory` and `list_memories` sent a rewritten path the same way | Refused with `GoodMemIdError`; nothing is sent |
| 262 of 300 hostile calls (30 call paths × 10 ids such as `../spaces/<id>`, `%2e%2e/spaces/<id>`, `<id>?x=1`, `<id>#frag`) reached the server, 244 of them returning a result | 0 of 300 |
| Tool id arguments were a bare `string` in the schema the model sees | A UUID pattern and description (DSPy 3.x; DSPy 2.5 shows the type name `UUIDStr`) |
| A lone `uuid.UUID` as `space_ids` (`retrieve`, `GoodMemRM`, `make_goodmem_tools`) raised `TypeError: 'UUID' object is not iterable` | Treated as one space |
| 55 offline + 19 live tests | 138 offline + 20 live |

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
| 62 tests patching `requests`, green against every defect; no CI | 55 offline + 19 live; CI on 3.10–3.13 |

## Tests

| Suite | Count | Needs |
| --- | --- | --- |
| `tests/test_goodmem_dspy.py` | 157 | nothing — the real SDK over a mock transport, fed NDJSON captured from a live server, plus a local recording HTTP server for the id checks |
| `tests/test_goodmem_dspy_live.py` | 23 | `GOODMEM_API_KEY` + `GOODMEM_BASE_URL`; skips entirely without them |

```bash
pip install -e . pytest httpx "ruff==0.7.4" mypy

pytest tests/test_goodmem_dspy.py

GOODMEM_API_KEY=... GOODMEM_BASE_URL=... \
  GOODMEM_TEST_EMBEDDER_ID=... GOODMEM_TEST_RERANKER_ID=... \
  pytest tests/test_goodmem_dspy_live.py

# what CI runs
ruff check src tests && ruff format --check src tests && mypy src/goodmem_dspy
```

The live suite creates one space per run and asserts, against a fresh server
listing, that it is gone afterwards.

## A note on TLS

`verify_ssl` exists for self-signed development servers and defaults to on.
0.1.1's README and the retriever's own docstring both showed it turned off, so
that is what people copied; no example here does, and CI fails if one appears.

## License

MIT.
