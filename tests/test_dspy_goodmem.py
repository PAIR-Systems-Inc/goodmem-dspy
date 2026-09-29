"""Offline tests for goodmem-dspy (import package ``dspy_goodmem``).

These drive the *real* GoodMem SDK over an ``httpx`` mock transport, fed with
NDJSON and JSON captured from a live GoodMem server (v1.0.320). 0.1.1's suite
patched ``requests.get``/``requests.post`` instead, which is the boundary the
defects lived behind: all 62 of its passing tests were green against every one
of them.

The id tests at the end go one step further: the production client, the SDK
and ``httpx`` talk over a real socket to a local server that records every
request, because the path an id produces is decided inside ``httpx``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import dspy
import httpx
import pytest

from dspy_goodmem import (
    GoodMemClient,
    GoodMemError,
    GoodMemRM,
    filters,
    make_goodmem_tools,
)
from dspy_goodmem._filters import GoodMemFilterError
from dspy_goodmem._results import (
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    classify_status,
    orient_score,
    outcome_from_events,
)
from dspy_goodmem._uploads import GoodMemUploadError, resolve_upload_path

FIXTURES = Path(__file__).parent / "goodmem_fixtures"
BASE = "https://goodmem.test"

# GoodMem ids are UUIDs, and 0.2.1 refuses anything else before a request is
# made, so the ids below are real ones captured with the fixtures.
SPACE = "01a0d44b-746f-775b-b91e-bc73d4058e27"
SPACE_2 = "01a0d44b-96ae-7081-bc16-5644e701222a"
MEMORY = "01a0d44b-748d-72eb-b54e-c3ea2d956927"
EMBEDDER_A = "019cfd1c-c033-7517-b7de-f73941a0464b"
EMBEDDER_B = "019e3d24-0763-70f5-9786-da6b30b90d2f"
RERANKER = "01a0d44c-5e1a-7a2b-9c3d-4e5f60718293"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def make_client(handler, **kwargs) -> GoodMemClient:
    from goodmem import Goodmem

    sdk = Goodmem(
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=BASE,
            headers={"X-API-Key": "gm_offline_test_key"},
        ),
    )
    return GoodMemClient(api_key="gm_offline_test_key", base_url=BASE, client=sdk, **kwargs)


def retrieve_handler(payload: bytes, *, capture: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":retrieve"):
            if capture is not None:
                capture["body"] = json.loads(request.content)
            return httpx.Response(200, content=payload, headers={"content-type": "application/x-ndjson"})
        return httpx.Response(404, json={"message": "unexpected"})

    return handler


class TestFixturesAreReal:
    def test_fixtures_are_real_server_bytes(self):
        stream = fixture("retrieve_ok.ndjson").decode()
        events = [json.loads(line) for line in stream.strip().split("\n") if line.strip()]
        assert any("resultSetBoundary" in e for e in events)
        assert any("retrievedItem" in e for e in events)
        assert re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-", stream, re.I)

    def test_no_credential_in_fixtures(self):
        for path in FIXTURES.iterdir():
            assert not re.search(rb"gm_[a-z0-9]{20,}", path.read_bytes())


class TestRetrievalStatusContract:
    def test_q4a_degraded_with_hits_returns_the_hits(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        outcome = c.retrieve("canary", [SPACE])
        assert len(outcome.hits) > 0, "hits were discarded"
        assert outcome.partial is True
        assert {"NOT_FOUND", "RERANKING_FAILED"} <= {s.code for s in outcome.statuses}

    def test_q4b_degraded_without_hits_is_flagged(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_empty.ndjson")))
        outcome = c.retrieve("nothing", [SPACE])
        assert outcome.hits == []
        assert outcome.partial is True and outcome.statuses

    def test_q1_informational_codes_are_noise(self):
        assert classify_status("FEATURE_DISABLED", "x").informational is True
        assert classify_status("LLM_CAPABILITY_INFERRED", "x").informational is True

    def test_q3_unknown_code_becomes_unknown_and_is_never_dropped(self):
        payload = json.dumps({"status": {"code": "FUTURE", "message": "new"}}).encode() + b"\n"
        c = make_client(retrieve_handler(payload))
        outcome = c.retrieve("q", [SPACE])
        assert [s.code for s in outcome.statuses] == [UNKNOWN_CODE]
        assert outcome.partial is True

    def test_a_clean_stream_is_not_partial(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        outcome = c.retrieve("canary", [SPACE])
        assert outcome.partial is False and outcome.statuses == []
        assert len(outcome.hits) >= 1

    def test_a_truncated_stream_keeps_what_arrived(self):
        whole = fixture("retrieve_ok.ndjson")
        c = make_client(retrieve_handler(whole[: int(len(whole) * 0.6)]))
        outcome = c.retrieve("canary", [SPACE])
        assert outcome.partial is True
        assert MALFORMED_STREAM_CODE in {s.code for s in outcome.statuses}


class TestRetrieverSurface:
    def test_passages_carry_more_than_long_text(self):
        """0.1.1 handed DSPy dotdict({'long_text': ...}) and nothing else."""
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        rm = GoodMemRM(space_ids=[SPACE], client=c, k=3)
        p = dict(rm("canary").passages[0])
        assert p["long_text"]
        for key in ("score", "raw_score", "score_kind", "chunk_id", "memory_id", "metadata"):
            assert key in p, f"{key} is not reaching DSPy"

    def test_scores_are_higher_is_better_with_the_raw_value_kept(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        p = dict(GoodMemRM(space_ids=[SPACE], client=c)("canary").passages[0])
        assert p["raw_score"] < 0 and p["score"] > 0
        assert p["score"] == pytest.approx(-p["raw_score"])
        assert p["score_kind"] == "vector"

    def test_a_degraded_retrieval_with_no_passages_warns(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_empty.ndjson")))
        rm = GoodMemRM(space_ids=[SPACE], client=c)
        with pytest.warns(UserWarning, match="not an empty index"):
            out = rm("nothing")
        assert out.passages == []

    def test_a_genuinely_empty_result_does_not_warn(self, recwarn):
        c = make_client(retrieve_handler(b""))
        out = GoodMemRM(space_ids=[SPACE], client=c)("nothing")
        assert out.passages == []
        assert not [w for w in recwarn if "not an empty index" in str(w.message)]

    def test_degraded_with_hits_returns_them_flagged(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        passages = GoodMemRM(space_ids=[SPACE], client=c)("canary").passages
        assert passages and dict(passages[0])["goodmem_partial"] is True

    def test_k_is_respected(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        assert len(GoodMemRM(space_ids=[SPACE], client=c, k=1)("canary").passages) <= 1

    def test_an_empty_space_list_is_refused(self):
        c = make_client(retrieve_handler(b""))
        with pytest.raises(ValueError, match="at least one space"):
            GoodMemRM(space_ids=[], client=c)

    def test_no_polling_knobs_remain_on_the_retriever(self):
        import inspect

        params = set(inspect.signature(GoodMemRM.__init__).parameters)
        for banned in ("wait_for_indexing", "poll_timeout", "poll_interval"):
            assert banned not in params

    def test_min_score_warns_and_names_the_range(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        rm = GoodMemRM(space_ids=[SPACE], client=c, reranker_id=RERANKER, min_score=99.0)
        with pytest.warns(UserWarning, match="observed scores ranged"):
            assert rm("canary").passages == []

    def test_no_threshold_is_sent_by_default(self):
        capture: dict = {}
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture))
        GoodMemRM(space_ids=[SPACE], client=c)("canary")
        assert "relevanceThreshold" not in json.dumps(capture["body"])


class TestToolSurface:
    def test_default_tools_are_a_search_and_a_write(self):
        """0.1.1 exposed eleven tools including delete_space."""
        c = make_client(retrieve_handler(b""))
        assert [t.__name__ for t in make_goodmem_tools(c, [SPACE])] == [
            "goodmem_search",
            "goodmem_remember",
        ]

    def test_admin_and_delete_are_opt_in(self):
        c = make_client(retrieve_handler(b""))
        names = [getattr(t, "__name__", "") for t in make_goodmem_tools(c, [SPACE])]
        for banned in ("delete_space", "delete_memory", "update_space", "create_space"):
            assert banned not in names

    def test_admin_adds_management_but_not_deletion(self):
        c = make_client(retrieve_handler(b""))
        names = [getattr(t, "__name__", "") for t in make_goodmem_tools(c, [SPACE], allow_admin=True)]
        assert "create_space" in names and "delete_space" not in names

    def test_delete_is_separate(self):
        c = make_client(retrieve_handler(b""))
        names = [getattr(t, "__name__", "") for t in make_goodmem_tools(c, [SPACE], allow_delete=True)]
        assert "delete_space" in names and "delete_memory" in names

    def test_the_model_never_chooses_a_space(self):
        import inspect

        c = make_client(retrieve_handler(b""))
        search = make_goodmem_tools(c, [SPACE])[0]
        assert set(inspect.signature(search).parameters) == {"query", "top_k"}

    def test_upload_requires_an_upload_dir(self):
        c = make_client(retrieve_handler(b""))
        with pytest.raises(ValueError, match="upload_dir"):
            make_goodmem_tools(c, [SPACE], allow_upload=True)

    def test_search_tool_reports_partial(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        out = make_goodmem_tools(c, [SPACE])[0]("canary")
        assert out["partial"] is True and out["statuses"] and out["warning"]


class TestPublicReadRemoved:
    def test_update_space_has_no_public_read(self):
        """0.1.1 sent publicRead and the server answers 400."""
        import inspect

        assert "public_read" not in inspect.signature(GoodMemClient.update_space).parameters

    def test_no_public_read_in_any_shipped_code_path(self):
        import ast

        import dspy_goodmem

        offenders = []
        for path in Path(dspy_goodmem.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    node.value = ""
            if "publicRead" in ast.unparse(tree) or "public_read" in ast.unparse(tree):
                offenders.append(path.name)
        assert offenders == []


class TestUploads:
    def test_paths_outside_the_upload_dir_are_refused(self, tmp_path):
        for bad in ("/etc/hostname", "../../etc/hostname"):
            with pytest.raises(GoodMemUploadError, match="outside the upload"):
                resolve_upload_path(bad, tmp_path)

    def test_symlink_escape_is_refused(self, tmp_path):
        os.symlink("/etc/hostname", tmp_path / "escape.txt")
        with pytest.raises(GoodMemUploadError, match="outside the upload"):
            resolve_upload_path("escape.txt", tmp_path)

    def test_a_file_inside_is_allowed(self, tmp_path):
        (tmp_path / "ok.txt").write_text("hi")
        assert resolve_upload_path("ok.txt", tmp_path).name == "ok.txt"

    def test_uploads_off_without_a_dir(self):
        with pytest.raises(GoodMemUploadError, match="disabled"):
            resolve_upload_path("/etc/hostname", None)

    def test_create_memory_refuses_a_host_path(self, tmp_path):
        c = make_client(retrieve_handler(b""), upload_dir=tmp_path)
        with pytest.raises(GoodMemUploadError):
            c.create_memory(SPACE, file_name="/etc/hostname")


class TestCreateMemoryConveniences:
    """0.1.1 accepted source/author/tags and folded them into metadata; a
    rewrite that silently dropped them broke the shipped RAG example."""

    def _capture(self):
        capture: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/memories" and request.method == "POST":
                capture["body"] = json.loads(request.content)
                return httpx.Response(201, json=json.loads(fixture("memory_get.json")))
            return httpx.Response(404, json={"message": "unexpected"})

        return capture, make_client(handler)

    def test_source_author_and_tags_fold_into_metadata(self):
        capture, c = self._capture()
        c.create_memory(SPACE, text_content="x", source="kb", author="me", tags="a, b")
        meta = capture["body"]["metadata"]
        assert meta["source"] == "kb" and meta["author"] == "me"
        assert meta["tags"] == ["a", "b"]

    def test_explicit_metadata_is_kept_alongside(self):
        capture, c = self._capture()
        c.create_memory(SPACE, text_content="x", source="kb", metadata={"k": "v"})
        assert capture["body"]["metadata"] == {"k": "v", "source": "kb"}

    def test_file_path_alias_is_still_confined(self, tmp_path):
        """The 0.1.1 argument name works, but it cannot read outside upload_dir."""
        _, c = self._capture()
        c.upload_dir = tmp_path
        with pytest.raises(GoodMemUploadError, match="outside the upload"):
            c.create_memory(SPACE, file_path="/etc/hostname")


class TestFilters:
    def test_apostrophes_are_backslash_escaped(self):
        assert filters.equals("n", "o'brien").endswith(r"'o\'brien'")

    def test_control_characters_are_refused(self):
        with pytest.raises(GoodMemFilterError, match="control characters"):
            filters.equals("f", "a\nb")

    def test_booleans_cast_as_boolean(self):
        assert filters.equals("a", True) == "CAST(val('$.a') AS BOOLEAN) = true"

    def test_unsafe_field_names_are_refused(self):
        with pytest.raises(GoodMemFilterError, match="field name"):
            filters.equals("a' OR '1", "x")

    def test_the_filter_reaches_the_request(self):
        capture: dict = {}
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture))
        c.retrieve("q", [SPACE], metadata_filter={"tenant": "acme"})
        assert capture["body"]["spaceKeys"][0]["filter"] == ("CAST(val('$.tenant') AS TEXT) = 'acme'")


class TestSpacesAndErrors:
    def _spaces(self, spaces):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/v1/spaces":
                return httpx.Response(200, json={"spaces": spaces})
            return httpx.Response(404, json={"message": "unexpected"})

        return handler

    def _space(self, space_id, name, embedders):
        import copy

        t = copy.deepcopy(json.loads(fixture("spaces_page1.json"))["spaces"][0])
        t["spaceId"], t["name"] = space_id, name
        et = t["spaceEmbedders"][0]
        t["spaceEmbedders"] = []
        for e in embedders:
            clone = copy.deepcopy(et)
            clone["embedderId"], clone["spaceId"] = e, space_id
            t["spaceEmbedders"].append(clone)
        return t

    def test_reuse_requires_a_matching_embedder(self):
        c = make_client(self._spaces([self._space(SPACE, "notes", [EMBEDDER_A])]))
        with pytest.raises(GoodMemError) as err:
            c.create_space("notes", EMBEDDER_B)
        assert EMBEDDER_A in str(err.value) and EMBEDDER_B in str(err.value)

    def test_reuse_with_a_matching_embedder_succeeds(self):
        c = make_client(self._spaces([self._space(SPACE, "notes", [EMBEDDER_A])]))
        assert c.create_space("notes", EMBEDDER_A)["reused"] is True

    def test_an_ambiguous_name_is_an_error(self):
        c = make_client(
            self._spaces([self._space(SPACE, "notes", [EMBEDDER_A]), self._space(SPACE_2, "notes", [EMBEDDER_A])])
        )
        with pytest.raises(GoodMemError, match="refusing to guess"):
            c.create_space("notes", EMBEDDER_A)

    def test_listing_follows_pagination(self):
        p1 = json.loads(fixture("spaces_page1.json"))
        p2 = json.loads(fixture("spaces_page2.json"))
        p2.pop("nextToken", None)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json=p1 if len(calls) == 1 else p2)

        c = make_client(handler)
        spaces = c.list_spaces()
        assert len(calls) == 2, "the second page was never requested"
        assert len(spaces) == len(p1["spaces"]) + len(p2["spaces"])

    def test_the_servers_message_reaches_the_caller(self):
        body = fixture("error_400.json")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, content=body, headers={"content-type": "application/json"})

        c = make_client(handler)
        with pytest.raises(GoodMemError) as err:
            c.list_spaces()
        assert "Invalid embedder ID format" in str(err.value)
        assert err.value.status_code == 400


class TestContentAndSecrets:
    def _handler(self, content: bytes, content_type: str, status: int = 200):
        memory = json.loads(fixture("memory_get.json"))
        memory["contentType"] = content_type

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/content"):
                return httpx.Response(status, content=content, headers={"content-type": content_type})
            return httpx.Response(200, json=memory)

        return handler

    def test_text_content_comes_back_as_text(self):
        c = make_client(self._handler(b"hello", "text/plain"))
        out = c.get_memory(MEMORY, include_content=True)
        assert out["content"] == "hello" and out["contentEncoding"] == "text"

    def test_binary_content_is_base64_and_serialisable(self):
        import base64

        pdf = b"%PDF-1.4\x00\xff"
        c = make_client(self._handler(pdf, "application/pdf"))
        out = c.get_memory(MEMORY, include_content=True)
        json.dumps(out)
        assert base64.b64decode(out["content"]) == pdf

    def test_a_failed_content_fetch_raises(self):
        c = make_client(self._handler(b'{"m":"gone"}', "application/json", 404))
        with pytest.raises(GoodMemError):
            c.get_memory(MEMORY, include_content=True)

    def test_the_api_key_is_not_in_repr(self):
        c = make_client(retrieve_handler(b""))
        assert "gm_offline_test_key" not in repr(c)

    def test_the_api_key_is_not_a_public_attribute(self):
        c = make_client(retrieve_handler(b""))
        public = {v for k, v in vars(c).items() if not k.startswith("_") and isinstance(v, str)}
        assert "gm_offline_test_key" not in public

    def test_an_injected_client_is_not_closed(self):
        c = make_client(retrieve_handler(b""))
        c.close()
        assert c._owns_client is False


# ---------------------------------------------------------------------------
# dspy.Retrieve with GoodMemRM configured as dspy.settings.rm
# ---------------------------------------------------------------------------

#: The fixture stream every test in this block replays: a live retrieval
#: (2026-09-29) whose two hits are vector-scored.
RETRIEVE_STREAM = "retrieve_bad_reranker.ndjson"
CANARY_TEXT = "The fixture canary is ORYX-2290. DSPy retriever audit.\n"


def _texts(payload: bytes) -> list[str]:
    out = []
    for line in payload.decode().splitlines():
        if line.strip() and "retrievedItem" in line:
            out.append(json.loads(line)["retrievedItem"]["chunk"]["chunk"]["chunkText"])
    return out


class TestDspyRetrieve:
    """README "Retrieve" configures the retriever as ``dspy.settings.rm``;
    ``dspy.Retrieve`` is the DSPy module that reads that setting. It calls
    ``rm(query, k=k)`` and reads ``long_text`` from each item it iterates.
    0.2.1 returned a ``dspy.Prediction``, which iterates as its key names, so
    every call raised ``AttributeError: 'str' object has no attribute
    'long_text'``."""

    def test_dspy_retrieve_works_with_the_rm_in_context(self):
        c = make_client(retrieve_handler(fixture(RETRIEVE_STREAM)))
        rm = GoodMemRM(space_ids=[SPACE], client=c, k=2)
        with dspy.context(rm=rm):
            out = dspy.Retrieve(k=2)("What is the fixture canary?")
        assert out.passages == _texts(fixture(RETRIEVE_STREAM))
        assert all(isinstance(t, str) for t in out.passages)

    def test_dspy_retrieve_works_with_the_rm_configured_as_the_readme_says(self):
        c = make_client(retrieve_handler(fixture(RETRIEVE_STREAM)))
        rm = GoodMemRM(space_ids=[SPACE], client=c, k=2)
        dspy.settings.configure(rm=rm)
        try:
            out = dspy.Retrieve(k=1)("What is the fixture canary?")
        finally:
            dspy.settings.configure(rm=None)
        assert out.passages == [CANARY_TEXT]

    def test_dspy_retrieve_passes_its_k_to_the_rm(self):
        capture: dict = {}
        c = make_client(retrieve_handler(fixture(RETRIEVE_STREAM), capture=capture))
        with dspy.context(rm=GoodMemRM(space_ids=[SPACE], client=c, k=5)):
            dspy.Retrieve(k=1)("q")
        assert capture["body"]["requestedSize"] == 1

    def test_iterating_the_rm_output_yields_passages(self):
        c = make_client(retrieve_handler(fixture(RETRIEVE_STREAM)))
        out = GoodMemRM(space_ids=[SPACE], client=c, k=2)("canary")
        assert [p.long_text for p in out] == _texts(fixture(RETRIEVE_STREAM))

    def test_passages_is_still_available_and_carries_the_metadata(self):
        c = make_client(retrieve_handler(fixture(RETRIEVE_STREAM)))
        out = GoodMemRM(space_ids=[SPACE], client=c, k=2)("canary")
        assert out.passages == list(out)
        p = out.passages[0]
        assert p.long_text == CANARY_TEXT
        assert p.chunk_id and p.memory_id and p.space_id and p.score_kind == "vector"
        assert p.score == pytest.approx(-p.raw_score)

    def test_the_result_is_a_list_that_carries_the_degraded_flag(self):
        from dspy_goodmem import GoodMemPassages

        c = make_client(retrieve_handler(fixture("retrieve_degraded_empty.ndjson")))
        with pytest.warns(UserWarning, match="not an empty index"):
            out = GoodMemRM(space_ids=[SPACE], client=c)("nothing")
        assert isinstance(out, GoodMemPassages) and isinstance(out, list)
        assert out == [] and out.partial is True
        assert {"NOT_FOUND", "RERANKING_FAILED"} <= {s["code"] for s in out.statuses}

    def test_a_clean_result_is_not_flagged(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        out = GoodMemRM(space_ids=[SPACE], client=c)("canary")
        assert out and out.partial is False and out.statuses == []


# ---------------------------------------------------------------------------
# A reranker that was requested but failed
# ---------------------------------------------------------------------------


def _lines(name: str) -> list[str]:
    return [line for line in fixture(name).decode().splitlines() if line.strip()]


def _stream(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode()


def _status_code(line: str) -> str | None:
    return json.loads(line).get("status", {}).get("code")


class TestRerankerFallback:
    """With a reranker requested that the server cannot run, GoodMem reports
    ``NOT_FOUND`` (naming ``reranker_id``) and ``RERANKING_FAILED`` and still
    returns the vector-stage hits, scored as vector scores. 0.2.1 labelled
    them from configuration: ``score_kind="reranker"`` with the raw negative
    value as ``score``, and ``min_score`` then discarded all of them. The
    fixture is that stream, captured live on 2026-09-29."""

    def test_fallback_passages_are_vector_scored_higher_is_better(self):
        c = make_client(retrieve_handler(fixture("retrieve_bad_reranker.ndjson")))
        passages = GoodMemRM(space_ids=[SPACE], client=c, k=2, reranker_id=RERANKER)("canary").passages
        assert len(passages) == 2
        for p in passages:
            assert p.raw_score < 0
            assert p.score_kind == "vector"
            assert p.score == pytest.approx(-p.raw_score)
            assert p.goodmem_partial is True
        scores = [p.score for p in passages]
        assert scores == sorted(scores, reverse=True)

    def test_the_search_tool_reports_fallback_hits_as_vector(self):
        c = make_client(retrieve_handler(fixture("retrieve_bad_reranker.ndjson")))
        search = make_goodmem_tools(c, [SPACE], reranker_id=RERANKER)[0]
        out = search("canary", 2)
        assert out["totalResults"] == 2
        assert all(h["scoreKind"] == "vector" and h["score"] > 0 for h in out["results"])
        assert out["partial"] is True and out["warning"]
        assert {"NOT_FOUND", "RERANKING_FAILED"} <= {s["code"] for s in out["statuses"]}

    def test_min_score_does_not_discard_fallback_hits(self, recwarn):
        """Q4a: problem + hits -> the hits. min_score is a reranker threshold;
        on 0.2.1 it removed both hits and then warned that the index was not
        empty."""
        c = make_client(retrieve_handler(fixture("retrieve_bad_reranker.ndjson")))
        rm = GoodMemRM(space_ids=[SPACE], client=c, k=2, reranker_id=RERANKER, min_score=0.9)
        out = rm("canary")
        assert len(out.passages) == 2
        assert out.partial is True
        assert not [w for w in recwarn if "min_score" in str(w.message)]
        assert not [w for w in recwarn if "not an empty index" in str(w.message)]

    def test_the_outcome_says_it_was_not_reranked(self):
        c = make_client(retrieve_handler(fixture("retrieve_bad_reranker.ndjson")))
        outcome = c.retrieve("canary", [SPACE], reranker_id=RERANKER)
        assert outcome.reranked is False
        assert outcome.partial is True
        assert [h.score_kind for h in outcome.hits] == ["vector", "vector"]

    def test_reranking_failed_after_the_hits_still_counts(self):
        """The decision is made once the whole stream is read."""
        lines = _lines("retrieve_bad_reranker.ndjson")
        failed = [line for line in lines if _status_code(line) == "RERANKING_FAILED"]
        rest = [line for line in lines if _status_code(line) not in ("RERANKING_FAILED", "NOT_FOUND")]
        c = make_client(retrieve_handler(_stream(rest + failed)))
        outcome = c.retrieve("canary", [SPACE], reranker_id=RERANKER)
        assert outcome.reranked is False
        assert [h.score_kind for h in outcome.hits] == ["vector", "vector"]
        assert all(h.score > 0 for h in outcome.hits)

    def test_a_not_found_naming_the_reranker_alone_counts(self):
        lines = [line for line in _lines("retrieve_bad_reranker.ndjson") if _status_code(line) != "RERANKING_FAILED"]
        c = make_client(retrieve_handler(_stream(lines)))
        outcome = c.retrieve("canary", [SPACE], reranker_id=RERANKER)
        assert [s.code for s in outcome.statuses] == ["NOT_FOUND"]
        assert outcome.reranked is False
        assert all(h.score_kind == "vector" and h.score > 0 for h in outcome.hits)

    def test_a_reranker_not_found_without_details_counts_by_its_message(self):
        lines = []
        for line in _lines("retrieve_bad_reranker.ndjson"):
            if _status_code(line) == "RERANKING_FAILED":
                continue
            if _status_code(line) == "NOT_FOUND":
                event = json.loads(line)
                del event["status"]["details"]
                line = json.dumps(event)
            lines.append(line)
        c = make_client(retrieve_handler(_stream(lines)))
        outcome = c.retrieve("canary", [SPACE], reranker_id=RERANKER)
        assert outcome.reranked is False

    def test_an_llm_not_found_leaves_reranker_scores_alone(self):
        """Live: a real reranker with a missing LLM. The server reports
        NOT_FOUND naming ``llm_id`` and SUMMARIZATION_FAILED; the hits were
        reranked (0.875, 0.25) and must stay labelled so."""
        c = make_client(retrieve_handler(fixture("retrieve_reranked_bad_llm.ndjson")))
        outcome = c.retrieve("canary", [SPACE], reranker_id=RERANKER)
        assert {s.code for s in outcome.statuses} == {"NOT_FOUND", "SUMMARIZATION_FAILED"}
        assert outcome.partial is True
        assert outcome.reranked is True
        assert [(h.score_kind, h.score) for h in outcome.hits] == [("reranker", 0.875), ("reranker", 0.25)]

    def test_an_unrelated_problem_keeps_reranker_scores(self):
        lines = _lines("retrieve_reranked_ok.ndjson")
        future = json.dumps({"status": {"code": "SOME_FUTURE_CODE", "message": "odd"}})
        c = make_client(retrieve_handler(_stream([future, *lines])))
        outcome = c.retrieve("canary", [SPACE], reranker_id=RERANKER)
        assert [s.code for s in outcome.statuses] == [UNKNOWN_CODE]
        assert outcome.reranked is True
        assert all(h.score_kind == "reranker" for h in outcome.hits)

    def test_a_working_reranker_is_reported_as_reranked(self):
        c = make_client(retrieve_handler(fixture("retrieve_reranked_ok.ndjson")))
        passages = GoodMemRM(space_ids=[SPACE], client=c, k=2, reranker_id=RERANKER)("canary").passages
        assert [(p.score_kind, p.score, p.raw_score) for p in passages] == [
            ("reranker", 0.875, 0.875),
            ("reranker", 0.25, 0.25),
        ]
        assert not any(p.goodmem_partial for p in passages)

    def test_min_score_still_applies_to_real_reranker_scores(self):
        c = make_client(retrieve_handler(fixture("retrieve_reranked_ok.ndjson")))
        rm = GoodMemRM(space_ids=[SPACE], client=c, k=2, reranker_id=RERANKER, min_score=0.5)
        assert [p.score for p in rm("canary").passages] == [0.875]

    def test_without_a_reranker_nothing_is_reranked(self):
        c = make_client(retrieve_handler(fixture("retrieve_reranked_ok.ndjson")))
        outcome = c.retrieve("canary", [SPACE])
        assert outcome.reranked is False


class TestScores:
    def test_vector_scores_flip(self):
        assert orient_score(-0.51, reranked=False) == pytest.approx(0.51)

    def test_reranker_scores_do_not_flip(self):
        assert orient_score(0.93, reranked=True) == pytest.approx(0.93)
        assert orient_score(-0.14, reranked=True) == pytest.approx(-0.14)


class TestJoin:
    def _chunk(self, chunk_id, text, memory_id, score):
        import copy

        for line in fixture("retrieve_ok.ndjson").decode().strip().split("\n"):
            e = json.loads(line)
            if "retrievedItem" in e:
                e = copy.deepcopy(e)
                ref = e["retrievedItem"]["chunk"]
                ref["relevanceScore"] = score
                ref["chunk"]["chunkId"] = chunk_id
                ref["chunk"]["chunkText"] = text
                ref["chunk"]["memoryId"] = memory_id
                return e
        raise AssertionError("no chunk event in the fixture")

    def _definition(self, memory_id, metadata):
        import copy

        for line in fixture("retrieve_ok.ndjson").decode().strip().split("\n"):
            e = json.loads(line)
            if "memoryDefinition" in e:
                e = copy.deepcopy(e)
                e["memoryDefinition"]["memoryId"] = memory_id
                e["memoryDefinition"]["metadata"] = metadata
                return e
        raise AssertionError("no definition event in the fixture")

    def _as_models(self, events):
        from goodmem.models import RetrieveMemoryEvent

        return [RetrieveMemoryEvent.model_validate(e) for e in events]

    def test_join_is_by_uuid_not_arrival_order(self):
        events = [
            self._chunk("c1", "alpha", "mem-A", -0.2),
            self._chunk("c2", "bravo", "mem-B", -0.4),
            self._definition("mem-B", {"tag": "B"}),
            self._definition("mem-A", {"tag": "A"}),
        ]
        out = outcome_from_events(self._as_models(events))
        by_id = {h.chunk_id: h for h in out.hits}
        assert by_id["c1"].metadata == {"tag": "A"}
        assert by_id["c2"].metadata == {"tag": "B"}

    def test_duplicate_chunks_collapse_but_distinct_ones_do_not(self):
        dup = outcome_from_events(
            self._as_models([self._chunk("c1", "a", "m", -0.2), self._chunk("c1", "a", "m", -0.2)])
        )
        assert len(dup.hits) == 1
        two = outcome_from_events(
            self._as_models([self._chunk("c1", "a", "m", -0.2), self._chunk("c2", "b", "m", -0.3)])
        )
        assert len(two.hits) == 2


# ---------------------------------------------------------------------------
# Ids reach URL paths, so anything that is not a UUID is refused
# ---------------------------------------------------------------------------

#: The official SDK builds paths as ``f"/v1/memories/{id}"`` and ``httpx``
#: resolves dot segments before sending, so on 0.2.0 every one of these
#: reached the server -- ``../spaces/<id>`` passed as a memory id arrived as
#: ``DELETE /v1/spaces/<id>`` and deleted the whole space.
HOSTILE_IDS = [
    f"../spaces/{SPACE}",
    f"a/../../spaces/{SPACE}",
    f"%2e%2e/spaces/{SPACE}",
    f"..%2Fspaces%2F{SPACE}",
    f"{SPACE}/../../spaces/{SPACE}",
    "",
    f" {SPACE}",
    f"{SPACE}?x=1",
    f"{SPACE}#frag",
    f"{SPACE}\n",
]

#: What the model is shown for every id argument (literal here so that this
#: file still imports against 0.2.0, where the tests below must fail).
UUID_SCHEMA_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _goodmem_response(method: str, path: str) -> tuple[int, str, bytes]:
    """What a GoodMem server answers, keyed on the route the request resolved to."""
    as_json = "application/json"
    space = json.dumps(json.loads(fixture("spaces_page1.json"))["spaces"][1]).encode()
    if method == "DELETE":
        return 204, "", b""
    if path == "/v1/memories:retrieve":
        return 200, "application/x-ndjson", fixture("retrieve_ok.ndjson")
    if path == "/v1/memories":
        return 200, as_json, fixture("memory_get.json")
    if path == "/v1/spaces":
        return (200, as_json, b'{"spaces": []}') if method == "GET" else (200, as_json, space)
    if path.startswith("/v1/spaces/"):
        return (200, as_json, b'{"memories": []}') if path.endswith("/memories") else (200, as_json, space)
    if path.startswith("/v1/memories/"):
        if path.endswith("/content"):
            return 200, "text/plain", b"hello"
        return 200, as_json, fixture("memory_get.json")
    return 404, as_json, b'{"message": "unexpected"}'


class RecordingServer:
    """A real HTTP server on 127.0.0.1 that records every request it receives.

    It answers the way GoodMem does -- ``204`` to any DELETE, the captured
    fixtures to anything else it recognises -- so a request that should never
    have been sent is not rescued by an error response: on 0.2.0,
    ``delete_memory("../spaces/<id>")`` came back ``{"success": True}``.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes]] = []
        recorded = self.requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def serve(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                recorded.append((self.command, self.path, body))
                status, content_type, payload = _goodmem_response(self.command, urlsplit(self.path).path)
                self.send_response(status)
                if content_type:
                    self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            setattr(Handler, f"do_{method}", Handler.serve)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def sent(self) -> list[str]:
        """Each recorded request as ``METHOD <request-target as received>``."""
        return [f"{method} {target}" for method, target, _ in self.requests]

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture(scope="module")
def recording_server():
    server = RecordingServer()
    yield server
    server.close()


@pytest.fixture
def wire(recording_server, monkeypatch):
    """The production client -- SDK and httpx included -- aimed at the recorder."""
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    client = GoodMemClient(api_key="gm_offline_test_key", base_url=recording_server.url, timeout=5)
    recording_server.requests.clear()
    yield recording_server, client
    client.close()


def assert_refused_before_any_request(server, call, field, *, expected=None):
    """Every hostile id must raise a ValueError naming ``field`` and send nothing."""
    expected = f"{field} must be a UUID" if expected is None else expected
    failures = []
    for hostile in HOSTILE_IDS:
        server.requests.clear()
        try:
            outcome = f"returned {call(hostile)!r}"
            refused = False
        except ValueError as exc:
            outcome = f"raised {type(exc).__name__}: {exc}"
            refused = expected in str(exc)
        except Exception as exc:
            outcome = f"raised {type(exc).__name__}: {exc}"
            refused = False
        if server.requests or not refused:
            failures.append(f"  {hostile!r}: the server received {server.sent or 'nothing'}; the call {outcome[:160]}")
    assert not failures, (
        f"{len(failures)} of {len(HOSTILE_IDS)} hostile {field} values were not refused before a request:\n"
        + "\n".join(failures)
    )


#: Client methods whose id the SDK interpolates into a URL path.
PATH_CALLS = {
    "GoodMemClient.get_memory": ("memory_id", lambda c, v: c.get_memory(v)),
    "GoodMemClient.get_memory(include_content=True)": (
        "memory_id",
        lambda c, v: c.get_memory(v, include_content=True),
    ),
    "GoodMemClient.list_memories": ("space_id", lambda c, v: c.list_memories(v)),
    "GoodMemClient.delete_memory": ("memory_id", lambda c, v: c.delete_memory(v)),
    "GoodMemClient.get_space": ("space_id", lambda c, v: c.get_space(v)),
    "GoodMemClient.update_space": ("space_id", lambda c, v: c.update_space(v, name="renamed")),
    "GoodMemClient.delete_space": ("space_id", lambda c, v: c.delete_space(v)),
}

#: Entry points whose ids travel in a request body. There is no path to
#: escape, but every id follows one rule, checked by the same helper.
BODY_CALLS = {
    "GoodMemClient.retrieve(space_ids)": ("space_ids[0]", lambda c, v: c.retrieve("q", [v])),
    "GoodMemClient.retrieve(reranker_id)": ("reranker_id", lambda c, v: c.retrieve("q", [SPACE], reranker_id=v)),
    "GoodMemClient.create_memory": ("space_id", lambda c, v: c.create_memory(v, text_content="x")),
    "GoodMemClient.create_space": ("embedder_id", lambda c, v: c.create_space("notes", v)),
    "GoodMemRM(space_ids)": ("space_ids[0]", lambda c, v: GoodMemRM(space_ids=[v], client=c)("q")),
    "GoodMemRM(reranker_id)": (
        "reranker_id",
        lambda c, v: GoodMemRM(space_ids=[SPACE], client=c, reranker_id=v)("q"),
    ),
    "make_goodmem_tools(space_ids) goodmem_search": ("space_ids[0]", lambda c, v: make_goodmem_tools(c, [v])[0]("q")),
    "make_goodmem_tools(space_ids) goodmem_remember": (
        "space_ids[0]",
        lambda c, v: make_goodmem_tools(c, [v])[1]("x"),
    ),
    "make_goodmem_tools(reranker_id) goodmem_search": (
        "reranker_id",
        lambda c, v: make_goodmem_tools(c, [SPACE], reranker_id=v)[0]("q"),
    ),
}

#: The id-taking tools make_goodmem_tools hands a model under allow_admin and
#: allow_delete: the id argument, and any other argument the call needs.
MODEL_TOOLS = {
    "get_memory": ("memory_id", {}),
    "list_memories": ("space_id", {}),
    "get_space": ("space_id", {}),
    "update_space": ("space_id", {"name": "renamed"}),
    "create_space": ("embedder_id", {"name": "notes"}),
    "delete_memory": ("memory_id", {}),
    "delete_space": ("space_id", {}),
}


def model_tool(client, name):
    tools = make_goodmem_tools(client, [SPACE], allow_admin=True, allow_delete=True)
    return {getattr(t, "__name__", ""): t for t in tools}[name]


class TestIdsMustBeUuids:
    """An id that reaches the SDK becomes part of a URL path, so it must be a UUID."""

    @pytest.mark.parametrize("name", list(PATH_CALLS))
    def test_a_path_id_that_is_not_a_uuid_never_reaches_the_server(self, wire, name):
        server, client = wire
        field, call = PATH_CALLS[name]
        assert_refused_before_any_request(server, lambda v: call(client, v), field)

    @pytest.mark.parametrize("name", list(BODY_CALLS))
    def test_a_body_id_that_is_not_a_uuid_is_refused_too(self, wire, name):
        server, client = wire
        field, call = BODY_CALLS[name]
        assert_refused_before_any_request(server, lambda v: call(client, v), field)

    @pytest.mark.parametrize("name", list(MODEL_TOOLS))
    def test_a_model_tool_refuses_a_hostile_id(self, wire, name):
        """Called directly: the client's own check, with no DSPy schema in front."""
        server, client = wire
        field, extra = MODEL_TOOLS[name]
        tool = model_tool(client, name)
        assert_refused_before_any_request(server, lambda v: tool(**{field: v}, **extra), field)

    @pytest.mark.parametrize("name", list(MODEL_TOOLS))
    def test_a_model_tool_refuses_a_hostile_id_through_dspy(self, wire, name):
        """As ReAct calls it. DSPy 3.x checks the schema first; DSPy 2.5 leaves it to the client."""
        server, client = wire
        field, extra = MODEL_TOOLS[name]
        tool = dspy.Tool(model_tool(client, name))
        # dspy.Tool on 3.x refuses in its own words (a jsonschema or pydantic
        # error, both ValueErrors), so only the type and the silence are checked.
        assert_refused_before_any_request(server, lambda v: tool(**{field: v}, **extra), field, expected="")

    @pytest.mark.parametrize("name", list(MODEL_TOOLS))
    def test_the_model_is_told_the_id_is_a_uuid(self, wire, name):
        _, client = wire
        field, _ = MODEL_TOOLS[name]
        schema = dspy.Tool(model_tool(client, name)).args[field]
        if isinstance(schema, str):
            # DSPy 2.5 shows the model a type name rather than a JSON schema.
            assert schema == "UUIDStr"
            return
        assert schema["type"] == "string"
        assert schema["pattern"] == UUID_SCHEMA_PATTERN
        assert "UUID" in schema["description"]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("GoodMemClient.get_memory", [f"GET /v1/memories/{MEMORY}"]),
            (
                "GoodMemClient.get_memory(include_content=True)",
                [f"GET /v1/memories/{MEMORY}", f"GET /v1/memories/{MEMORY}/content"],
            ),
            ("GoodMemClient.list_memories", [f"GET /v1/spaces/{SPACE}/memories"]),
            ("GoodMemClient.delete_memory", [f"DELETE /v1/memories/{MEMORY}"]),
            ("GoodMemClient.get_space", [f"GET /v1/spaces/{SPACE}"]),
            ("GoodMemClient.update_space", [f"PUT /v1/spaces/{SPACE}"]),
            ("GoodMemClient.delete_space", [f"DELETE /v1/spaces/{SPACE}"]),
        ],
    )
    def test_a_uuid_reaches_exactly_the_intended_path(self, wire, name, expected):
        server, client = wire
        field, call = PATH_CALLS[name]
        call(client, MEMORY if field == "memory_id" else SPACE)
        assert [f"{method} {urlsplit(target).path}" for method, target, _ in server.requests] == expected

    def test_a_model_tool_with_a_uuid_still_works_through_dspy(self, wire):
        server, client = wire
        out = dspy.Tool(model_tool(client, "delete_memory"))(memory_id=MEMORY)
        assert out == {"success": True, "memoryId": MEMORY}
        assert server.sent == [f"DELETE /v1/memories/{MEMORY}"]

    def test_body_ids_arrive_intact(self, wire):
        server, client = wire
        client.retrieve("q", [SPACE], reranker_id=RERANKER)
        ((method, target, body),) = server.requests
        assert (method, urlsplit(target).path) == ("POST", "/v1/memories:retrieve")
        sent = json.loads(body)
        assert [key["spaceId"] for key in sent["spaceKeys"]] == [SPACE]
        assert sent["postProcessor"]["config"]["reranker_id"] == RERANKER

    def test_create_memory_sends_the_space_it_was_given(self, wire):
        server, client = wire
        client.create_memory(SPACE, text_content="x")
        ((method, target, body),) = server.requests
        assert (method, urlsplit(target).path) == ("POST", "/v1/memories")
        assert json.loads(body)["spaceId"] == SPACE

    def test_create_space_sends_the_embedder_it_was_given(self, wire):
        server, client = wire
        client.create_space("notes", EMBEDDER_A)
        assert [f"{method} {urlsplit(target).path}" for method, target, _ in server.requests] == [
            "GET /v1/spaces",
            "POST /v1/spaces",
        ]
        assert json.loads(server.requests[1][2])["spaceEmbedders"][0]["embedderId"] == EMBEDDER_A

    def test_an_uppercase_uuid_is_sent_lowercase(self, wire):
        server, client = wire
        assert client.delete_memory(MEMORY.upper()) == {"success": True, "memoryId": MEMORY}
        assert server.sent == [f"DELETE /v1/memories/{MEMORY}"]

    def test_a_uuid_object_is_accepted(self, wire):
        server, client = wire
        client.delete_space(uuid.UUID(SPACE))
        assert server.sent == [f"DELETE /v1/spaces/{SPACE}"]

    def test_the_call_boundary_holds_even_if_configuration_changes_later(self, wire):
        """GoodMemRM checks its ids when built; the client checks them again on every call."""
        server, client = wire
        rm = GoodMemRM(space_ids=[SPACE], client=client)
        rm.space_ids = [f"../spaces/{SPACE}"]
        with pytest.raises(ValueError, match=r"space_ids\[0\] must be a UUID"):
            rm("q")
        assert server.requests == []

    def test_only_the_canonical_form_is_accepted(self):
        """``uuid.UUID()`` parses braces, URNs and unhyphenated hex; an id may not."""
        import dspy_goodmem
        from dspy_goodmem._ids import require_uuid

        assert require_uuid(SPACE.upper(), "space_id") == SPACE
        assert require_uuid(uuid.UUID(SPACE), "space_id") == SPACE
        for bad in (None, 123, SPACE.encode(), SPACE.replace("-", ""), "{" + SPACE + "}", f"urn:uuid:{SPACE}"):
            with pytest.raises(dspy_goodmem.GoodMemIdError, match="space_id must be a UUID"):
                require_uuid(bad, "space_id")

    def test_the_refusal_is_a_value_error_exported_from_the_package(self):
        import dspy_goodmem

        assert issubclass(dspy_goodmem.GoodMemIdError, ValueError)
        assert "GoodMemIdError" in dspy_goodmem.__all__


# What a caller's own object may not do to an id after it has been checked
# ---------------------------------------------------------------------------


class _RewritingStr(str):
    """A ``str`` whose methods lie: every derived value is a traversal.

    Only Python code can build one -- model JSON and configuration strings
    cannot -- but on the first 0.2.1 draft the check ran on the real
    characters and the id sent was ``value.lower()``, which this overrides.
    """

    def lower(self):  # type: ignore[override]
        return f"../spaces/{SPACE}"

    def __str__(self):
        return f"../spaces/{SPACE}"

    def __format__(self, spec):
        return f"../spaces/{SPACE}"


class _RewritingUUID(uuid.UUID):
    """A ``uuid.UUID`` whose string form is a traversal."""

    def __str__(self):
        return f"../spaces/{SPACE}"


class _PretendStr:
    """Not a ``str`` at all, though ``isinstance(x, str)`` says it is."""

    @property  # type: ignore[misc]
    def __class__(self):
        return str

    def lower(self):
        return f"../spaces/{SPACE}"


#: The id each path call is given, and the one request it must produce.
PATH_TARGETS = {
    "GoodMemClient.get_memory": (MEMORY, [f"GET /v1/memories/{MEMORY}"]),
    "GoodMemClient.get_memory(include_content=True)": (
        MEMORY,
        [f"GET /v1/memories/{MEMORY}", f"GET /v1/memories/{MEMORY}/content"],
    ),
    "GoodMemClient.list_memories": (SPACE_2, [f"GET /v1/spaces/{SPACE_2}/memories"]),
    "GoodMemClient.delete_memory": (MEMORY, [f"DELETE /v1/memories/{MEMORY}"]),
    "GoodMemClient.get_space": (SPACE_2, [f"GET /v1/spaces/{SPACE_2}"]),
    "GoodMemClient.update_space": (SPACE_2, [f"PUT /v1/spaces/{SPACE_2}"]),
    "GoodMemClient.delete_space": (SPACE_2, [f"DELETE /v1/spaces/{SPACE_2}"]),
}


def _paths(server) -> list[str]:
    return [f"{method} {urlsplit(target).path}" for method, target, _ in server.requests]


class TestIdObjectsCannotRewriteThemselves:
    """The id sent is the one that was checked, whatever the caller's object does."""

    @pytest.mark.parametrize("name", list(PATH_CALLS))
    def test_a_str_subclass_reaches_only_the_id_it_holds(self, wire, name):
        server, client = wire
        _, call = PATH_CALLS[name]
        valid, expected = PATH_TARGETS[name]
        try:
            call(client, _RewritingStr(valid.upper()))
        except GoodMemError:
            pass  # the recorder's fixture body may not parse; only the path matters
        assert _paths(server) == expected

    @pytest.mark.parametrize("name", list(PATH_CALLS))
    def test_a_uuid_subclass_whose_text_is_not_a_uuid_is_refused(self, wire, name):
        server, client = wire
        field, call = PATH_CALLS[name]
        valid, _ = PATH_TARGETS[name]
        with pytest.raises(ValueError, match=f"{field} must be a UUID"):
            call(client, _RewritingUUID(valid))
        assert server.requests == []

    @pytest.mark.parametrize("name", list(PATH_CALLS))
    def test_an_object_pretending_to_be_a_str_is_refused(self, wire, name):
        server, client = wire
        field, call = PATH_CALLS[name]
        with pytest.raises(ValueError, match=f"{field} must be a UUID"):
            call(client, _PretendStr())
        assert server.requests == []

    def test_a_str_subclass_in_a_body_is_sent_as_the_id_it_holds(self, wire):
        server, client = wire
        client.retrieve("q", [_RewritingStr(SPACE)], reranker_id=_RewritingStr(RERANKER))
        client.create_memory(_RewritingStr(SPACE), text_content="x")
        retrieve, create = (json.loads(body) for _, _, body in server.requests)
        assert [key["spaceId"] for key in retrieve["spaceKeys"]] == [SPACE]
        assert retrieve["postProcessor"]["config"]["reranker_id"] == RERANKER
        assert create["spaceId"] == SPACE

    def test_the_validator_returns_a_plain_lowercase_str(self):
        from dspy_goodmem._ids import require_uuid

        checked = require_uuid(_RewritingStr(SPACE.upper()), "space_id")
        assert type(checked) is str
        assert checked == SPACE


# Configured ids are refused when the retriever or tools are built
# ---------------------------------------------------------------------------

#: Construction only -- nothing is called afterwards, so the check each
#: client call makes cannot stand in for the one made at startup.
BUILD_CALLS = {
    "GoodMemRM(space_ids=[id])": ("space_ids[0]", lambda c, v: GoodMemRM(space_ids=[v], client=c)),
    "GoodMemRM(space_ids=id)": ("space_ids[0]", lambda c, v: GoodMemRM(space_ids=v, client=c)),
    "GoodMemRM(reranker_id)": ("reranker_id", lambda c, v: GoodMemRM(space_ids=[SPACE], client=c, reranker_id=v)),
    "make_goodmem_tools(space_ids=[id])": ("space_ids[0]", lambda c, v: make_goodmem_tools(c, [v])),
    "make_goodmem_tools(space_ids=id)": ("space_ids[0]", lambda c, v: make_goodmem_tools(c, v)),
    "make_goodmem_tools(reranker_id)": (
        "reranker_id",
        lambda c, v: make_goodmem_tools(c, [SPACE], reranker_id=v),
    ),
}


class TestConfiguredIdsFailAtStartup:
    @pytest.mark.parametrize("name", list(BUILD_CALLS))
    def test_a_configured_id_that_is_not_a_uuid_fails_when_built(self, wire, name):
        server, client = wire
        field, build = BUILD_CALLS[name]
        assert_refused_before_any_request(server, lambda v: build(client, v), field)

    def test_a_single_uuid_object_is_accepted_as_the_space(self, wire):
        """A lone ``uuid.UUID`` is one space, as a lone string is -- not an iterable to unpack."""
        server, client = wire
        space = uuid.UUID(SPACE)
        client.retrieve("q", space)
        GoodMemRM(space_ids=space, client=client)("q")
        make_goodmem_tools(client, space)[0]("q")
        assert _paths(server) == ["POST /v1/memories:retrieve"] * 3
        for _, _, body in server.requests:
            assert [key["spaceId"] for key in json.loads(body)["spaceKeys"]] == [SPACE]
