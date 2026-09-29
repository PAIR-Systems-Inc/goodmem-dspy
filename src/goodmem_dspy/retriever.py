"""GoodMem retriever for DSPy.

Provides :class:`GoodMemRM`, a ``dspy.Retrieve`` subclass that performs
semantic retrieval against one or more GoodMem spaces.

Example::

    import dspy
    from goodmem_dspy import GoodMemRM

    rm = GoodMemRM(space_ids=["<space-uuid>"], k=3)
    passages = rm("What is the main finding?").passages

    dspy.settings.configure(rm=rm)
    texts = dspy.Retrieve(k=3)("What is the main finding?").passages

Each passage is a ``dotdict`` carrying ``long_text`` -- what DSPy consumes --
alongside the identifiers and score that 0.1.1 discarded, so a program can
cite, filter or de-duplicate what it retrieved.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import Any

import dspy

from goodmem_dspy._dotdict import dotdict
from goodmem_dspy._ids import id_list, require_uuid, require_uuids
from goodmem_dspy.client import GoodMemClient

logger = logging.getLogger(__name__)


class GoodMemPassages(list[dotdict]):
    """The passages one :class:`GoodMemRM` call returned.

    A plain list of passage ``dotdict`` objects -- the shape DSPy's own
    retrievers return and the one ``dspy.Retrieve`` iterates, reading
    ``long_text`` from each -- that also answers ``.passages`` with itself,
    so ``rm(query).passages`` works as well.

    0.2.0 and 0.2.1 returned a ``dspy.Prediction`` instead. Iterating a
    ``Prediction`` yields its key names, so with the retriever configured as
    ``dspy.settings.rm`` every ``dspy.Retrieve`` call raised
    ``AttributeError: 'str' object has no attribute 'long_text'``.

    Attributes:
        partial: ``True`` when the server reported a problem during this
            retrieval -- including when no passage came back, which a bare
            empty list could not otherwise tell apart from an empty index.
        statuses: The problems the server reported, as plain dictionaries.
    """

    def __init__(
        self,
        passages: Iterable[dotdict] = (),
        *,
        partial: bool = False,
        statuses: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(passages)
        self.partial = partial
        self.statuses = list(statuses or [])

    @property
    def passages(self) -> GoodMemPassages:
        """The passages themselves, as ``dspy.Prediction.passages`` would be."""
        return self


class GoodMemRM(dspy.Retrieve):
    """Retrieve passages from GoodMem spaces.

    Args:
        space_ids: One space id, or several. Each must be a UUID.
        api_key: GoodMem API key. Falls back to ``GOODMEM_API_KEY``.
        base_url: GoodMem server URL. Falls back to ``GOODMEM_BASE_URL``.
        k: How many passages to return.
        verify_ssl: Whether to verify TLS certificates. Leave this on; it
            exists for self-signed development servers only.
        timeout: Per-request timeout in seconds.
        reranker_id: A reranker to apply to retrieval, by UUID.
        min_score: Drop passages scoring below this value. Applies only to
            reranker scores, because reranker scales are provider-dependent:
            not without ``reranker_id``, and not when the server reports that
            reranking failed and returns vector hits instead. Off by default.
        metadata_filter: Metadata every retrieved memory must match, applied
            server-side.
        client: An already-configured :class:`GoodMemClient` to reuse.

    Raises:
        ValueError: If no space id is given.
        GoodMemIdError: If a space id or the reranker id is not a UUID --
            checked here so a misconfiguration fails at startup, and again
            by the client on every retrieval.
    """

    def __init__(
        self,
        space_ids: list[str] | str,
        api_key: str | None = None,
        base_url: str | None = None,
        k: int = 3,
        *,
        verify_ssl: bool = True,
        timeout: float | None = 30.0,
        reranker_id: str | None = None,
        min_score: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        client: GoodMemClient | None = None,
    ) -> None:
        super().__init__(k=k)
        self.space_ids = id_list(space_ids)
        if not self.space_ids:
            raise ValueError("GoodMemRM needs at least one space id.")
        self.space_ids = require_uuids(self.space_ids, "space_ids")
        self.reranker_id = require_uuid(reranker_id, "reranker_id") if reranker_id is not None else None
        self.min_score = min_score
        self.metadata_filter = dict(metadata_filter or {})
        self._owns_client = client is None
        self.client = client or GoodMemClient(
            api_key=api_key,
            base_url=base_url,
            verify_ssl=verify_ssl,
            timeout=timeout,
        )

    def forward(self, query_or_queries: str | list[str], k: int | None = None) -> GoodMemPassages:
        """Retrieve passages for one query or several.

        A retrieval the server reported a problem for still returns whatever
        passages arrived, each flagged ``goodmem_partial``. A degraded
        retrieval that returns nothing comes back empty with ``partial`` set,
        and also raises a ``UserWarning`` and logs at WARNING with the
        server's own reason -- rather than looking like a clean miss.

        Args:
            query_or_queries: The query, or a list of queries.
            k: Override the configured number of passages.

        Returns:
            A :class:`GoodMemPassages` list of passages, which ``dspy.Retrieve``
            consumes directly and which also answers ``.passages``.
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
        queries = [q for q in queries if q]
        limit = k if k is not None else self.k

        passages: list[dotdict] = []
        seen: set[str] = set()
        degraded: list[dict[str, Any]] = []

        for query in queries:
            outcome = self.client.retrieve(
                query,
                self.space_ids,
                max_results=limit,
                reranker_id=self.reranker_id,
                metadata_filter=self.metadata_filter or None,
            )
            if outcome.partial:
                degraded.extend(outcome.status_dicts)

            hits = outcome.hits
            # A reranker threshold applies only to reranker scores. When the
            # reranker failed, the server returns vector hits instead; the
            # threshold would discard what the server returned (Q4a).
            if self.min_score is not None and outcome.reranked:
                kept = [h for h in hits if h.score is not None and h.score >= self.min_score]
                if hits and not kept:
                    observed = [h.score for h in hits if h.score is not None]
                    warnings.warn(
                        f"min_score={self.min_score} removed all {len(hits)} "
                        f"reranked passage(s); observed scores ranged "
                        f"{min(observed):.4f}..{max(observed):.4f}. Reranker "
                        "scales are provider-dependent, not 0-1.",
                        UserWarning,
                        stacklevel=2,
                    )
                hits = kept

            for hit in hits:
                if hit.chunk_id in seen:
                    continue
                seen.add(hit.chunk_id)
                passages.append(
                    dotdict(
                        {
                            "long_text": hit.text,
                            "score": hit.score,
                            "raw_score": hit.raw_score,
                            "score_kind": hit.score_kind,
                            "chunk_id": hit.chunk_id,
                            "memory_id": hit.memory_id,
                            "space_id": hit.space_id,
                            "metadata": hit.metadata,
                            "goodmem_partial": outcome.partial,
                        }
                    )
                )

        if degraded and not passages:
            detail = "; ".join(f"{s['code']}: {s.get('message', '')}" for s in degraded)
            message = (
                "GoodMem reported a problem during retrieval and returned no "
                f"passages -- this is not an empty index: {detail}"
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            logger.warning(message)

        return GoodMemPassages(
            passages[:limit] if limit else passages,
            partial=bool(degraded),
            statuses=degraded,
        )
