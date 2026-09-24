"""Offline tests for dspy-goodmem.

These drive the *real* GoodMem SDK over an ``httpx`` mock transport, fed with
NDJSON and JSON captured from a live GoodMem server (v1.0.320). 0.1.1's suite
patched ``requests.get``/``requests.post`` instead, which is the boundary the
defects lived behind: all 62 of its passing tests were green against every one
of them.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

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
        outcome = c.retrieve("canary", ["s-1"])
        assert len(outcome.hits) > 0, "hits were discarded"
        assert outcome.partial is True
        assert {"NOT_FOUND", "RERANKING_FAILED"} <= {s.code for s in outcome.statuses}

    def test_q4b_degraded_without_hits_is_flagged(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_empty.ndjson")))
        outcome = c.retrieve("nothing", ["s-1"])
        assert outcome.hits == []
        assert outcome.partial is True and outcome.statuses

    def test_q1_informational_codes_are_noise(self):
        assert classify_status("FEATURE_DISABLED", "x").informational is True
        assert classify_status("LLM_CAPABILITY_INFERRED", "x").informational is True

    def test_q3_unknown_code_becomes_unknown_and_is_never_dropped(self):
        payload = json.dumps({"status": {"code": "FUTURE", "message": "new"}}).encode() + b"\n"
        c = make_client(retrieve_handler(payload))
        outcome = c.retrieve("q", ["s-1"])
        assert [s.code for s in outcome.statuses] == [UNKNOWN_CODE]
        assert outcome.partial is True

    def test_a_clean_stream_is_not_partial(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        outcome = c.retrieve("canary", ["s-1"])
        assert outcome.partial is False and outcome.statuses == []
        assert len(outcome.hits) >= 1

    def test_a_truncated_stream_keeps_what_arrived(self):
        whole = fixture("retrieve_ok.ndjson")
        c = make_client(retrieve_handler(whole[: int(len(whole) * 0.6)]))
        outcome = c.retrieve("canary", ["s-1"])
        assert outcome.partial is True
        assert MALFORMED_STREAM_CODE in {s.code for s in outcome.statuses}


class TestRetrieverSurface:
    def test_passages_carry_more_than_long_text(self):
        """0.1.1 handed DSPy dotdict({'long_text': ...}) and nothing else."""
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        rm = GoodMemRM(space_ids=["s-1"], client=c, k=3)
        p = dict(rm("canary").passages[0])
        assert p["long_text"]
        for key in ("score", "raw_score", "score_kind", "chunk_id", "memory_id", "metadata"):
            assert key in p, f"{key} is not reaching DSPy"

    def test_scores_are_higher_is_better_with_the_raw_value_kept(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        p = dict(GoodMemRM(space_ids=["s-1"], client=c)("canary").passages[0])
        assert p["raw_score"] < 0 and p["score"] > 0
        assert p["score"] == pytest.approx(-p["raw_score"])
        assert p["score_kind"] == "vector"

    def test_a_degraded_retrieval_with_no_passages_warns(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_empty.ndjson")))
        rm = GoodMemRM(space_ids=["s-1"], client=c)
        with pytest.warns(UserWarning, match="not an empty index"):
            out = rm("nothing")
        assert out.passages == []

    def test_a_genuinely_empty_result_does_not_warn(self, recwarn):
        c = make_client(retrieve_handler(b""))
        out = GoodMemRM(space_ids=["s-1"], client=c)("nothing")
        assert out.passages == []
        assert not [w for w in recwarn if "not an empty index" in str(w.message)]

    def test_degraded_with_hits_returns_them_flagged(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        passages = GoodMemRM(space_ids=["s-1"], client=c)("canary").passages
        assert passages and dict(passages[0])["goodmem_partial"] is True

    def test_k_is_respected(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        assert len(GoodMemRM(space_ids=["s-1"], client=c, k=1)("canary").passages) <= 1

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
        rm = GoodMemRM(space_ids=["s-1"], client=c, reranker_id="rr", min_score=99.0)
        with pytest.warns(UserWarning, match="observed scores ranged"):
            assert rm("canary").passages == []

    def test_no_threshold_is_sent_by_default(self):
        capture: dict = {}
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture))
        GoodMemRM(space_ids=["s-1"], client=c)("canary")
        assert "relevanceThreshold" not in json.dumps(capture["body"])


class TestToolSurface:
    def test_default_tools_are_a_search_and_a_write(self):
        """0.1.1 exposed eleven tools including delete_space."""
        c = make_client(retrieve_handler(b""))
        assert [t.__name__ for t in make_goodmem_tools(c, ["s-1"])] == [
            "goodmem_search",
            "goodmem_remember",
        ]

    def test_admin_and_delete_are_opt_in(self):
        c = make_client(retrieve_handler(b""))
        names = [getattr(t, "__name__", "") for t in make_goodmem_tools(c, ["s-1"])]
        for banned in ("delete_space", "delete_memory", "update_space", "create_space"):
            assert banned not in names

    def test_admin_adds_management_but_not_deletion(self):
        c = make_client(retrieve_handler(b""))
        names = [getattr(t, "__name__", "") for t in make_goodmem_tools(c, ["s-1"], allow_admin=True)]
        assert "create_space" in names and "delete_space" not in names

    def test_delete_is_separate(self):
        c = make_client(retrieve_handler(b""))
        names = [getattr(t, "__name__", "") for t in make_goodmem_tools(c, ["s-1"], allow_delete=True)]
        assert "delete_space" in names and "delete_memory" in names

    def test_the_model_never_chooses_a_space(self):
        import inspect

        c = make_client(retrieve_handler(b""))
        search = make_goodmem_tools(c, ["s-1"])[0]
        assert set(inspect.signature(search).parameters) == {"query", "top_k"}

    def test_upload_requires_an_upload_dir(self):
        c = make_client(retrieve_handler(b""))
        with pytest.raises(ValueError, match="upload_dir"):
            make_goodmem_tools(c, ["s-1"], allow_upload=True)

    def test_search_tool_reports_partial(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        out = make_goodmem_tools(c, ["s-1"])[0]("canary")
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
            c.create_memory("s-1", file_name="/etc/hostname")


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
        c.create_memory("s-1", text_content="x", source="kb", author="me", tags="a, b")
        meta = capture["body"]["metadata"]
        assert meta["source"] == "kb" and meta["author"] == "me"
        assert meta["tags"] == ["a", "b"]

    def test_explicit_metadata_is_kept_alongside(self):
        capture, c = self._capture()
        c.create_memory("s-1", text_content="x", source="kb", metadata={"k": "v"})
        assert capture["body"]["metadata"] == {"k": "v", "source": "kb"}

    def test_file_path_alias_is_still_confined(self, tmp_path):
        """The 0.1.1 argument name works, but it cannot read outside upload_dir."""
        _, c = self._capture()
        c.upload_dir = tmp_path
        with pytest.raises(GoodMemUploadError, match="outside the upload"):
            c.create_memory("s-1", file_path="/etc/hostname")


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
        c.retrieve("q", ["s-1"], metadata_filter={"tenant": "acme"})
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
        c = make_client(self._spaces([self._space("s-1", "notes", ["emb-a"])]))
        with pytest.raises(GoodMemError) as err:
            c.create_space("notes", "emb-b")
        assert "emb-a" in str(err.value) and "emb-b" in str(err.value)

    def test_reuse_with_a_matching_embedder_succeeds(self):
        c = make_client(self._spaces([self._space("s-1", "notes", ["emb-a"])]))
        assert c.create_space("notes", "emb-a")["reused"] is True

    def test_an_ambiguous_name_is_an_error(self):
        c = make_client(self._spaces([self._space("s-1", "notes", ["emb-a"]), self._space("s-2", "notes", ["emb-a"])]))
        with pytest.raises(GoodMemError, match="refusing to guess"):
            c.create_space("notes", "emb-a")

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
        out = c.get_memory("m-1", include_content=True)
        assert out["content"] == "hello" and out["contentEncoding"] == "text"

    def test_binary_content_is_base64_and_serialisable(self):
        import base64

        pdf = b"%PDF-1.4\x00\xff"
        c = make_client(self._handler(pdf, "application/pdf"))
        out = c.get_memory("m-1", include_content=True)
        json.dumps(out)
        assert base64.b64decode(out["content"]) == pdf

    def test_a_failed_content_fetch_raises(self):
        c = make_client(self._handler(b'{"m":"gone"}', "application/json", 404))
        with pytest.raises(GoodMemError):
            c.get_memory("m-1", include_content=True)

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
