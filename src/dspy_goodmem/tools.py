"""GoodMem tools for DSPy agents.

0.1.1 handed a ReAct agent eleven tools, including ``delete_space``, a
``public_read`` argument the server rejects, and an unrestricted
``file_path``. A model does not need to administer a memory server in order
to use one, so the default surface here is a search and a write; everything
else is opt-in and chosen by the developer.
"""

from __future__ import annotations

from typing import Any, Callable

from dspy_goodmem.client import GoodMemClient


def make_goodmem_tools(
    client: GoodMemClient,
    space_ids: list[str] | str,
    *,
    allow_write: bool = True,
    allow_admin: bool = False,
    allow_delete: bool = False,
    allow_upload: bool = False,
    reranker_id: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Callable[..., Any]]:
    """Build the GoodMem tools a DSPy agent may call.

    Args:
        client: A configured :class:`GoodMemClient`.
        space_ids: The space or spaces the agent may read and write. The
            model never chooses a space.
        allow_write: Whether the agent may store new memories.
        allow_admin: Whether space and embedder management is exposed.
        allow_delete: Whether the agent may delete memories and spaces.
        allow_upload: Whether the agent may upload files. Requires the client
            to have been given an ``upload_dir``.
        reranker_id: A reranker applied to the agent's searches.
        metadata_filter: Metadata every retrieved memory must match.

    Returns:
        Plain functions, ready for ``dspy.Tool`` or ``dspy.ReAct``.

    Raises:
        ValueError: If uploads are requested without an ``upload_dir``.
    """
    ids = [space_ids] if isinstance(space_ids, str) else list(space_ids)
    if not ids:
        raise ValueError("make_goodmem_tools() needs at least one space id.")
    if allow_upload and client.upload_dir is None:
        raise ValueError(
            "allow_upload=True requires the client to be constructed with "
            "upload_dir=<directory>; without one no path is ever read."
        )

    def goodmem_search(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search stored memories for information relevant to a question.

        Args:
            query: A natural-language description of what to find.
            top_k: How many results to return.

        Returns:
            ``results`` (matching chunks with their text and metadata),
            ``partial`` (True when the server reported a problem during this
            search), ``statuses``, and ``query``.
        """
        outcome = client.retrieve(
            query,
            ids,
            max_results=top_k,
            reranker_id=reranker_id,
            metadata_filter=metadata_filter or None,
        )
        result: dict[str, Any] = {
            "success": True,
            "query": query,
            "results": [h.as_dict() for h in outcome.hits],
            "totalResults": len(outcome.hits),
            "partial": outcome.partial,
            "statuses": outcome.status_dicts,
        }
        if outcome.partial:
            result["warning"] = outcome.warning_text()
        return result

    def goodmem_remember(text: str) -> dict[str, Any]:
        """Store a piece of text as a memory for later recall.

        Args:
            text: The text to remember.

        Returns:
            ``success``, ``memoryId`` and ``spaceId``.
        """
        return client.create_memory(ids[0], text_content=text)

    def goodmem_upload_file(file_name: str) -> dict[str, Any]:
        """Store a file from the configured upload directory as a memory.

        Args:
            file_name: The name of a file inside the upload directory. Any
                other path is refused.

        Returns:
            ``success``, ``memoryId`` and ``spaceId``.
        """
        return client.create_memory(ids[0], file_name=file_name)

    tools: list[Callable[..., Any]] = [goodmem_search]
    if allow_write:
        tools.append(goodmem_remember)
    if allow_upload:
        tools.append(goodmem_upload_file)
    if allow_admin:
        tools.extend(
            [
                client.list_spaces,
                client.list_embedders,
                client.get_space,
                client.create_space,
                client.update_space,
                client.list_memories,
                client.get_memory,
            ]
        )
    if allow_delete:
        tools.extend([client.delete_memory, client.delete_space])
    return tools


__all__ = ["make_goodmem_tools"]
