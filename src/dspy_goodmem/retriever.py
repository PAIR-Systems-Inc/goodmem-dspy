"""GoodMem retriever for DSPy.

Provides :class:`GoodMemRM`, a ``dspy.Retrieve`` subclass that performs
semantic retrieval against one or more GoodMem spaces.

Example::

    from dspy_goodmem import GoodMemRM

    rm = GoodMemRM(space_ids=["<space-uuid>"], k=3)
    passages = rm("What is the main finding?")

Each passage is a ``dotdict`` carrying ``long_text`` -- what DSPy consumes --
alongside the identifiers and score that 0.1.1 discarded, so a program can
cite, filter or de-duplicate what it retrieved.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import dspy

from dspy_goodmem._dotdict import dotdict
from dspy_goodmem.client import GoodMemClient

logger = logging.getLogger(__name__)


class GoodMemRM(dspy.Retrieve):
    """Retrieve passages from GoodMem spaces.

    Args:
        space_ids: One space id, or several.
        api_key: GoodMem API key. Falls back to ``GOODMEM_API_KEY``.
        base_url: GoodMem server URL. Falls back to ``GOODMEM_BASE_URL``.
        k: How many passages to return.
        verify_ssl: Whether to verify TLS certificates. Leave this on; it
            exists for self-signed development servers only.
        timeout: Per-request timeout in seconds.
        reranker_id: A reranker to apply to retrieval.
        min_score: Drop passages scoring below this value. Applies only with
            a reranker configured, because reranker scales are
            provider-dependent. Off by default.
        metadata_filter: Metadata every retrieved memory must match, applied
            server-side.
        client: An already-configured :class:`GoodMemClient` to reuse.
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
        self.space_ids = [space_ids] if isinstance(space_ids, str) else list(space_ids)
        if not self.space_ids:
            raise ValueError("GoodMemRM needs at least one space id.")
        self.reranker_id = reranker_id
        self.min_score = min_score
        self.metadata_filter = dict(metadata_filter or {})
        self._owns_client = client is None
        self.client = client or GoodMemClient(
            api_key=api_key,
            base_url=base_url,
            verify_ssl=verify_ssl,
            timeout=timeout,
        )

    def forward(self, query_or_queries: str | list[str], k: int | None = None) -> dspy.Prediction:
        """Retrieve passages for one query or several.

        A retrieval the server reported a problem for still returns whatever
        passages arrived; a DSPy ``Prediction`` has no slot for a flag on an
        empty result, so a degraded retrieval that returns nothing raises a
        ``UserWarning`` and logs at WARNING with the server's own reason --
        rather than looking like a clean miss.

        Args:
            query_or_queries: The query, or a list of queries.
            k: Override the configured number of passages.

        Returns:
            ``dspy.Prediction`` with ``passages``.
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
            if self.min_score is not None and self.reranker_id:
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

        return dspy.Prediction(passages=passages[:limit] if limit else passages)
