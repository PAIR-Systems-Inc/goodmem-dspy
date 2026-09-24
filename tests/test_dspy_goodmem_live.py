"""Live tests for dspy-goodmem, against a running GoodMem server.

These skip entirely unless ``GOODMEM_API_KEY`` and ``GOODMEM_BASE_URL`` are
set, which is also the check that no credential is baked into the package.
"""

from __future__ import annotations

import os
import time
import uuid
import warnings

import pytest

from dspy_goodmem import GoodMemClient, GoodMemError, GoodMemRM, make_goodmem_tools
from dspy_goodmem._uploads import GoodMemUploadError

API_KEY = os.environ.get("GOODMEM_API_KEY")
BASE_URL = os.environ.get("GOODMEM_BASE_URL")
VERIFY_SSL = os.environ.get("GOODMEM_VERIFY_SSL", "false").lower() == "true"
FAILING_EMBEDDER = os.environ.get("GOODMEM_TEST_FAILING_EMBEDDER_ID")

pytestmark = pytest.mark.skipif(
    not (API_KEY and BASE_URL),
    reason="GOODMEM_API_KEY and GOODMEM_BASE_URL are not set",
)

RUN = uuid.uuid4().hex[:8]


def _embedder_id(client: GoodMemClient) -> str:
    pinned = os.environ.get("GOODMEM_TEST_EMBEDDER_ID")
    if pinned:
        return pinned
    embedders = client.list_embedders()
    assert embedders, "the server has no embedders configured"
    return embedders[0]["embedderId"]


def _other_embedder_id(client: GoodMemClient) -> str:
    first = _embedder_id(client)
    others = [e["embedderId"] for e in client.list_embedders() if e["embedderId"] != first]
    if not others:
        pytest.skip("need two embedders to test a mismatch")
    return others[0]


@pytest.fixture(scope="module")
def client() -> GoodMemClient:
    c = GoodMemClient(verify_ssl=VERIFY_SSL)
    yield c
    c.close()


@pytest.fixture(scope="module")
def space(client: GoodMemClient):
    created = client.create_space(f"dspy-live-{RUN}", _embedder_id(client))
    space_id = created["spaceId"]
    yield space_id
    client.delete_space(space_id)
    assert not [s for s in client.list_spaces() if s["spaceId"] == space_id], "the space survived teardown"


@pytest.fixture(scope="module")
def seeded(client: GoodMemClient, space: str):
    canary = f"ORYX-{RUN.upper()}"
    created = client.create_memory(
        space, text_content=f"The DSPy live canary is {canary}.", metadata={"tenant": "acme"}
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if client.retrieve(canary, [space], max_results=3).hits:
            break
        time.sleep(2)
    else:
        pytest.fail("the seeded memory never became searchable")
    yield canary, created["memoryId"]
    client.delete_memory(created["memoryId"])


class TestLiveRetriever:
    def test_an_exact_identifier_round_trips(self, client, space, seeded):
        canary, memory_id = seeded
        passages = GoodMemRM(space_ids=[space], client=client, k=5)(canary).passages
        assert any(canary in p["long_text"] for p in passages)
        assert passages[0]["memory_id"] == memory_id

    def test_passages_carry_score_and_metadata(self, client, space, seeded):
        p = dict(GoodMemRM(space_ids=[space], client=client, k=1)(seeded[0]).passages[0])
        assert p["raw_score"] < 0, "GoodMem vector scores are negative"
        assert p["score"] > 0, "not flipped to higher-is-better"
        assert p["score_kind"] == "vector"
        assert p["metadata"]["tenant"] == "acme"
        assert p["chunk_id"] and p["memory_id"] and p["space_id"]

    def test_a_search_never_reaches_a_space_it_was_not_given(self, client, space, seeded):
        """The negative control is another space, not another query.

        Vector search returns nearest neighbours whatever the query, so
        searching for nonsense proves nothing. What must hold is that a
        retriever scoped to one space cannot see another's content.
        """
        other = client.create_space(f"dspy-live-other-{RUN}", _embedder_id(client))
        secret = f"ADDAX-{uuid.uuid4().hex[:6].upper()}"
        created = client.create_memory(other["spaceId"], text_content=f"Other {secret}.")
        try:
            deadline = time.time() + 60
            while time.time() < deadline:
                if client.retrieve(secret, [other["spaceId"]], max_results=3).hits:
                    break
                time.sleep(2)
            else:
                pytest.fail("the outsider memory never became searchable")

            scoped = GoodMemRM(space_ids=[space], client=client, k=10)(secret)
            assert all(secret not in p["long_text"] for p in scoped.passages)
            assert all(p["space_id"] != other["spaceId"] for p in scoped.passages)
        finally:
            client.delete_memory(created["memoryId"])
            client.delete_space(other["spaceId"])

    def test_the_read_path_does_not_poll(self, client):
        empty = client.create_space(f"dspy-live-fast-{RUN}", _embedder_id(client))
        try:
            started = time.time()
            out = GoodMemRM(space_ids=[empty["spaceId"]], client=client)("nothing at all")
            elapsed = time.time() - started
        finally:
            client.delete_space(empty["spaceId"])
        assert out.passages == []
        assert elapsed < 3.0, f"an empty search took {elapsed:.1f}s"


class TestLiveStatusContract:
    def test_a_failing_embedder_is_reported_not_silently_empty(self, client):
        """The sharpest 0.1.1 defect: EMBEDDER_FAILED looked like an empty index."""
        if not FAILING_EMBEDDER:
            pytest.skip("GOODMEM_TEST_FAILING_EMBEDDER_ID is not set")
        bad = client.create_space(f"dspy-live-bad-{RUN}", FAILING_EMBEDDER)
        created = None
        try:
            created = client.create_memory(bad["spaceId"], text_content="Doomed canary.")
            time.sleep(6)
            outcome = client.retrieve("doomed canary", [bad["spaceId"]], max_results=3)
            assert outcome.partial is True
            assert outcome.statuses, "the server's status was dropped"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                GoodMemRM(space_ids=[bad["spaceId"]], client=client)("doomed canary")
            assert any("not an empty index" in str(w.message) for w in caught)
        finally:
            if created:
                client.delete_memory(created["memoryId"])
            client.delete_space(bad["spaceId"])

    def test_a_healthy_search_reports_no_status(self, client, space, seeded):
        outcome = client.retrieve(seeded[0], [space], max_results=3)
        assert outcome.partial is False and outcome.statuses == []


class TestLiveFilters:
    def test_a_matching_filter_finds_it(self, client, space, seeded):
        out = client.retrieve(seeded[0], [space], max_results=5, metadata_filter={"tenant": "acme"})
        assert out.hits

    def test_an_injection_payload_matches_nothing(self, client, space, seeded):
        out = client.retrieve(seeded[0], [space], max_results=5, metadata_filter={"tenant": "x' OR '1'='1"})
        assert out.hits == [], "filter injection succeeded"

    def test_an_apostrophe_is_accepted_not_a_400(self, client, space):
        out = client.retrieve("anything", [space], max_results=1, metadata_filter={"tenant": "o'brien"})
        assert out.hits == []


class TestLiveSpaces:
    def test_publicread_is_gone_and_rename_works(self, client, space):
        renamed = f"dspy-live-{RUN}-renamed"
        assert client.update_space(space, name=renamed)["success"] is True
        assert client.get_space(space)["name"] == renamed
        client.update_space(space, name=f"dspy-live-{RUN}")

    def test_reuse_with_another_embedder_is_refused(self, client, space):
        with pytest.raises(GoodMemError, match="cannot be changed"):
            client.create_space(f"dspy-live-{RUN}", _other_embedder_id(client))

    def test_a_rejected_create_carries_the_servers_message(self, client):
        with pytest.raises(GoodMemError) as err:
            client.create_space(f"dspy-live-bad-{RUN}", "not-a-uuid")
        assert err.value.status_code == 400
        assert "embedder" in str(err.value).lower()

    def test_listing_is_paginated_and_unique(self, client):
        spaces = client.list_spaces()
        assert len({s["spaceId"] for s in spaces}) == len(spaces)
        assert any(s["name"].startswith(f"dspy-live-{RUN}") for s in spaces)


class TestLiveContentAndUploads:
    def test_text_content_round_trips(self, client, seeded):
        out = client.get_memory(seeded[1], include_content=True)
        assert out["contentEncoding"] == "text" and seeded[0] in out["content"]

    def test_binary_content_round_trips_as_base64(self, client, space, tmp_path):
        import base64
        import json as _json

        pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
        (tmp_path / "s.pdf").write_bytes(pdf)
        uploader = GoodMemClient(verify_ssl=VERIFY_SSL, upload_dir=tmp_path)
        created = uploader.create_memory(space, file_name="s.pdf")
        try:
            out = client.get_memory(created["memoryId"], include_content=True)
            _json.dumps(out)
            assert base64.b64decode(out["content"]) == pdf
        finally:
            client.delete_memory(created["memoryId"])
            uploader.close()

    def test_a_host_path_is_refused(self, client, space, tmp_path):
        uploader = GoodMemClient(verify_ssl=VERIFY_SSL, upload_dir=tmp_path)
        try:
            with pytest.raises(GoodMemUploadError):
                uploader.create_memory(space, file_name="/etc/hostname")
        finally:
            uploader.close()

    def test_uploads_are_off_without_an_upload_dir(self, client, space):
        with pytest.raises(GoodMemUploadError, match="disabled"):
            client.create_memory(space, file_name="anything.txt")


class TestLiveTools:
    def test_the_search_tool_works_end_to_end(self, client, space, seeded):
        search = make_goodmem_tools(client, [space])[0]
        out = search(seeded[0], 3)
        assert out["totalResults"] >= 1 and out["partial"] is False
        assert out["results"][0]["score"] > 0

    def test_the_write_tool_stores_and_recalls(self, client, space):
        tools = make_goodmem_tools(client, [space])
        search, remember = tools[0], tools[1]
        token = f"IBEX-{uuid.uuid4().hex[:6].upper()}"
        created = remember(f"Tool canary {token}.")
        try:
            deadline = time.time() + 60
            while time.time() < deadline:
                if search(token, 3)["totalResults"]:
                    break
                time.sleep(2)
            else:
                pytest.fail("the stored memory never became searchable")
        finally:
            client.delete_memory(created["memoryId"])
