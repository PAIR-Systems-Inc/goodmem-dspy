"""GoodMem client for DSPy, built on the official ``goodmem`` SDK.

0.1.1 wrapped ``requests`` by hand. This module keeps the same shape -- one
object holding the connection -- but delegates transport, pagination, stream
decoding and error typing to the SDK, and adds the retrieval status contract
on top.
"""

from __future__ import annotations

import base64
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from dspy_goodmem._filters import from_mapping
from dspy_goodmem._results import (
    RetrievalOutcome,
    log_if_degraded,
    outcome_from_events,
)
from dspy_goodmem._uploads import GoodMemUploadError, resolve_upload_path

logger = logging.getLogger(__name__)

DEFAULT_MAX_LIST_ITEMS = 200
DEFAULT_TIMEOUT = 30.0


class GoodMemError(RuntimeError):
    """Raised when a GoodMem operation fails.

    Attributes:
        status_code: The HTTP status the server returned, when the failure
            came from the server.
        body: The server's response body, verbatim.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _wrap(exc: Exception, what: str) -> GoodMemError:
    """Convert an SDK error into a GoodMemError, keeping the server's body."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = str(exc)
    if body and body not in detail:
        detail = f"{detail} -- {body}"
    return GoodMemError(f"{what} failed: {detail}", status_code=status, body=body)


def _space_embedder_ids(space: Any) -> list[str]:
    """Return the embedder ids a space is actually indexed by."""
    out: list[str] = []
    for config in getattr(space, "space_embedders", None) or []:
        value = getattr(config, "embedder_id", None)
        if value is None and isinstance(config, dict):
            value = config.get("embedderId") or config.get("embedder_id")
        if value:
            out.append(str(value))
    return out


def _decode_content(raw: bytes, content_type: str) -> tuple[Any, str]:
    """Decode memory content according to its content type."""
    primary = (content_type or "").split(";")[0].strip().lower()
    charset = "utf-8"
    for part in (content_type or "").split(";")[1:]:
        if "charset=" in part:
            charset = part.split("charset=", 1)[1].strip() or "utf-8"
    textual = primary.startswith("text/") or primary in {
        "application/json",
        "application/xml",
        "application/javascript",
    }
    if textual:
        try:
            return raw.decode(charset), "text"
        except (UnicodeDecodeError, LookupError):
            return base64.b64encode(raw).decode("ascii"), "base64"
    return base64.b64encode(raw).decode("ascii"), "base64"


class GoodMemClient:
    """A connection to a GoodMem server.

    Args:
        api_key: The GoodMem API key. Falls back to ``GOODMEM_API_KEY``.
        base_url: The server URL. Falls back to ``GOODMEM_BASE_URL``.
        verify_ssl: Whether to verify TLS certificates. Leave this on; it
            exists for self-signed development servers only.
        timeout: Per-request timeout in seconds, applied to every call.
        upload_dir: A directory that file uploads are confined to. When
            ``None``, no path is ever read from disk.
        max_list_items: Upper bound on items returned by a listing.
        client: An already-configured ``goodmem.Goodmem`` client. When given,
            its server, credentials and TLS settings are used as-is and it is
            never closed here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float | None = DEFAULT_TIMEOUT,
        upload_dir: str | Path | None = None,
        max_list_items: int = DEFAULT_MAX_LIST_ITEMS,
        client: Any = None,
    ) -> None:
        from goodmem import Goodmem

        self.base_url = (base_url or os.environ.get("GOODMEM_BASE_URL", "")).rstrip("/")
        resolved_key = api_key or os.environ.get("GOODMEM_API_KEY", "")
        # Held privately: never an attribute a repr or a config dump picks up.
        self.__api_key = resolved_key
        self.verify_ssl = verify_ssl
        self.max_list_items = max_list_items
        self.upload_dir = Path(upload_dir).expanduser().resolve() if upload_dir is not None else None

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            missing = [
                name
                for name, value in (
                    ("GOODMEM_API_KEY", resolved_key),
                    ("GOODMEM_BASE_URL", self.base_url),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"Missing GoodMem credentials: {', '.join(missing)}. Set "
                    "them in the environment or pass api_key/base_url, or "
                    "pass an already-configured client=Goodmem(...)."
                )
            self._client = Goodmem(
                base_url=self.base_url,
                api_key=resolved_key,
                timeout=timeout,
                verify=verify_ssl,
            )
            self._owns_client = True

    def __repr__(self) -> str:
        """A representation that never carries the API key."""
        return f"{type(self).__name__}(base_url={self.base_url!r})"

    def close(self) -> None:
        """Close the HTTP client, if this object created it."""
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> GoodMemClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        space_ids: str | list[str],
        *,
        max_results: int = 5,
        reranker_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalOutcome:
        """Retrieve chunks relevant to a query.

        Args:
            query: The natural-language query.
            space_ids: One space id or several.
            max_results: How many chunks to ask the server for.
            reranker_id: A reranker to apply, if any.
            metadata_filter: Metadata every memory must match, applied
                server-side and escaped by :mod:`dspy_goodmem.filters`.

        Returns:
            The hits and any statuses the server reported.
        """
        ids = [space_ids] if isinstance(space_ids, str) else list(space_ids)
        if not ids:
            raise GoodMemError("At least one space id is required.")
        expression = from_mapping(metadata_filter or {})
        keys: list[dict[str, Any]] = []
        for space_id in ids:
            key: dict[str, Any] = {"spaceId": space_id}
            if expression:
                key["filter"] = expression
            keys.append(key)

        kwargs: dict[str, Any] = {
            "message": query,
            "space_keys": keys,
            "requested_size": max_results,
            "fetch_memory": True,
        }
        if reranker_id:
            kwargs["reranker_id"] = reranker_id
        try:
            stream = self._client.memories.retrieve(**kwargs)
            with stream as events:
                outcome = outcome_from_events(events, reranked=bool(reranker_id))
        except GoodMemError:
            raise
        except Exception as exc:
            raise _wrap(exc, "Retrieval") from exc
        log_if_degraded(outcome, "goodmem retrieve")
        return outcome

    # ------------------------------------------------------------------
    # memories
    # ------------------------------------------------------------------

    def create_memory(
        self,
        space_id: str,
        *,
        text_content: str | None = None,
        file_name: str | None = None,
        file_path: str | None = None,
        source: str | None = None,
        author: str | None = None,
        tags: str | list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a memory from text, or from a file inside ``upload_dir``.

        ``source``, ``author`` and ``tags`` are folded into the memory's
        metadata, as 0.1.1 did. ``file_path`` is accepted as an alias of
        ``file_name`` for 0.1.1 callers, but it is confined to ``upload_dir``
        exactly the same way -- the unrestricted read is gone.

        Args:
            space_id: The space to write to.
            text_content: Text to store.
            file_name: A file inside the configured upload directory.
            file_path: Alias of ``file_name``; same confinement.
            source: Stored as ``metadata.source``.
            author: Stored as ``metadata.author``.
            tags: One tag, a comma-separated string, or a list; stored as
                ``metadata.tags``.
            metadata: Further key-value labels to attach.

        Returns:
            ``success``, ``memoryId``, ``spaceId`` and ``status``.
        """
        file_name = file_name or file_path
        if not text_content and not file_name:
            raise GoodMemError("Provide text_content or file_name.")
        meta: dict[str, Any] = dict(metadata or {})
        if source:
            meta["source"] = source
        if author:
            meta["author"] = author
        if tags:
            meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()] if isinstance(tags, str) else list(tags)
        kwargs: dict[str, Any] = {"space_id": space_id, "metadata": meta or None}
        if file_name:
            kwargs["file_path"] = str(resolve_upload_path(file_name, self.upload_dir))
        else:
            kwargs["original_content"] = text_content
            kwargs["content_type"] = "text/plain"
        try:
            memory = self._client.memories.create(**kwargs)
        except GoodMemUploadError:
            raise
        except Exception as exc:
            raise _wrap(exc, "Creating a memory") from exc
        return {
            "success": True,
            "memoryId": str(getattr(memory, "memory_id", "") or ""),
            "spaceId": str(getattr(memory, "space_id", "") or space_id),
            "status": str(getattr(memory, "processing_status", "") or ""),
        }

    def get_memory(self, memory_id: str, *, include_content: bool = False) -> dict[str, Any]:
        """Fetch one memory, optionally with its original content.

        Content is decoded by the memory's own content type: text as text,
        anything else as base64, so the result is always JSON-serialisable.
        A content fetch that fails is an error, not a successful result with
        a note in it.
        """
        try:
            memory = self._client.memories.get(id=memory_id)
        except Exception as exc:
            raise _wrap(exc, f"Fetching memory {memory_id}") from exc
        dump = getattr(memory, "model_dump", None)
        payload = dump(by_alias=True, exclude_none=True) if dump else {}
        result: dict[str, Any] = {"success": True, "memory": payload}
        if include_content:
            try:
                raw = self._client.memories.content(id=memory_id)
            except Exception as exc:
                raise _wrap(exc, f"Fetching content of memory {memory_id}") from exc
            content_type = str(payload.get("contentType") or payload.get("content_type") or "")
            result["content"], result["contentEncoding"] = _decode_content(raw, content_type)
        return result

    def list_memories(self, space_id: str) -> list[dict[str, Any]]:
        """List memories in a space, following pagination."""
        try:
            page = self._client.memories.list(space_id=space_id, max_items=self.max_list_items)
            memories = list(page)
        except Exception as exc:
            raise _wrap(exc, "Listing memories") from exc
        return [
            {
                "memoryId": str(getattr(m, "memory_id", "") or ""),
                "spaceId": str(getattr(m, "space_id", "") or ""),
                "contentType": str(getattr(m, "content_type", "") or ""),
                "processingStatus": str(getattr(m, "processing_status", "") or ""),
                "metadata": dict(getattr(m, "metadata", None) or {}),
            }
            for m in memories
        ]

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Permanently delete a memory and everything derived from it."""
        try:
            self._client.memories.delete(id=memory_id)
        except Exception as exc:
            raise _wrap(exc, f"Deleting memory {memory_id}") from exc
        return {"success": True, "memoryId": memory_id}

    # ------------------------------------------------------------------
    # spaces
    # ------------------------------------------------------------------

    def list_spaces(self) -> list[dict[str, Any]]:
        """List spaces, following pagination up to ``max_list_items``."""
        try:
            spaces = list(self._client.spaces.list(max_items=self.max_list_items))
        except Exception as exc:
            raise _wrap(exc, "Listing spaces") from exc
        return [
            {
                "spaceId": str(getattr(s, "space_id", "") or ""),
                "name": str(getattr(s, "name", "") or ""),
                "embedderIds": _space_embedder_ids(s),
            }
            for s in spaces
        ]

    def list_embedders(self) -> list[dict[str, Any]]:
        """List the embedder models available on the server."""
        try:
            embedders = list(self._client.embedders.list())
        except Exception as exc:
            raise _wrap(exc, "Listing embedders") from exc
        return [
            {
                "embedderId": str(getattr(e, "embedder_id", "") or ""),
                "displayName": str(getattr(e, "display_name", "") or ""),
                "modelIdentifier": str(getattr(e, "model_identifier", "") or ""),
            }
            for e in embedders
        ]

    def create_space(self, name: str, embedder_id: str) -> dict[str, Any]:
        """Create a space, or reuse one whose embedder already matches.

        A space cannot change embedder after creation, so reusing by name
        alone silently writes vectors from a different model than the caller
        asked for. Reuse requires a match; a mismatch names both.
        """
        try:
            existing = [
                s
                for s in self._client.spaces.list(max_items=self.max_list_items)
                if str(getattr(s, "name", "") or "") == name
            ]
        except Exception as exc:
            raise _wrap(exc, "Listing spaces") from exc
        if len(existing) > 1:
            raise GoodMemError(
                f"{len(existing)} spaces are named {name!r}; refusing to guess "
                "which one was meant. Pass a space id instead."
            )
        if existing:
            space = existing[0]
            actual = _space_embedder_ids(space)
            if embedder_id not in actual:
                raise GoodMemError(
                    f"Space {name!r} already exists and is indexed by "
                    f"embedder(s) {actual}, not {embedder_id!r}. An embedder "
                    "cannot be changed after creation."
                )
            return {
                "success": True,
                "spaceId": str(getattr(space, "space_id", "") or ""),
                "name": name,
                "embedderId": embedder_id,
                "reused": True,
            }
        try:
            space = self._client.spaces.create(
                name=name,
                space_embedders=[{"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}],
            )
        except Exception as exc:
            raise _wrap(exc, f"Creating space {name!r}") from exc
        return {
            "success": True,
            "spaceId": str(getattr(space, "space_id", "") or ""),
            "name": str(getattr(space, "name", "") or name),
            "embedderId": embedder_id,
            "reused": False,
        }

    def update_space(
        self,
        space_id: str,
        *,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        replace_labels: bool = False,
    ) -> dict[str, Any]:
        """Rename a space or edit its labels.

        ``publicRead`` is deliberately not offered: the server removed the
        field and rejects any request carrying it with HTTP 400.
        """
        request: dict[str, Any] = {}
        if name is not None:
            request["name"] = name
        if labels is not None:
            request["replaceLabels" if replace_labels else "mergeLabels"] = dict(labels)
        if not request:
            raise GoodMemError("update_space() needs a name or labels to change.")
        try:
            space = self._client.spaces.update(id=space_id, request=request)
        except Exception as exc:
            raise _wrap(exc, f"Updating space {space_id}") from exc
        return {
            "success": True,
            "spaceId": str(getattr(space, "space_id", "") or space_id),
            "name": str(getattr(space, "name", "") or ""),
        }

    def get_space(self, space_id: str) -> dict[str, Any]:
        """Fetch one space by id."""
        try:
            space = self._client.spaces.get(id=space_id)
        except Exception as exc:
            raise _wrap(exc, f"Fetching space {space_id}") from exc
        return {
            "success": True,
            "spaceId": str(getattr(space, "space_id", "") or ""),
            "name": str(getattr(space, "name", "") or ""),
            "embedderIds": _space_embedder_ids(space),
            "labels": dict(getattr(space, "labels", None) or {}),
        }

    def delete_space(self, space_id: str) -> dict[str, Any]:
        """Permanently delete a space and every memory in it."""
        try:
            self._client.spaces.delete(id=space_id)
        except Exception as exc:
            raise _wrap(exc, f"Deleting space {space_id}") from exc
        return {"success": True, "spaceId": space_id}


__all__ = ["GoodMemClient", "GoodMemError", "GoodMemUploadError", "warnings"]
