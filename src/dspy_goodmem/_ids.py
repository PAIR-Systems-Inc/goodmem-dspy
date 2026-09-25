r"""Refuses any GoodMem id that is not a canonical UUID.

The official ``goodmem`` SDK builds request paths by interpolating ids as
they are -- ``f"/v1/memories/{id}"`` -- and ``httpx`` resolves dot segments
before sending. On 0.2.0, ``delete_memory("../spaces/<space-id>")`` went out
as ``DELETE /v1/spaces/<space-id>`` and came back ``success: True``; a ``?``
or ``#`` in an id rewrote the query or was cut off, and the GoodMem server
decodes ``%2e%2e`` into ``..`` itself, so neither the client's encoding nor
the server can be relied on to contain an id.

Every GoodMem id -- space, memory, embedder, reranker -- is a UUID, so
anything else is refused here, before any request is built. This module is
the only place in the package that decides what an id may be.
"""

import re
import uuid
from collections.abc import Iterable
from typing import Annotated, Any

from pydantic import Field
from typing_extensions import TypeAliasType

#: A canonical UUID, in either case. Matched whole with ``fullmatch`` -- a
#: bare ``$`` would also accept a trailing newline.
UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

_UUID = re.compile(UUID_PATTERN)

_EXAMPLE = "01a0d44b-748d-72eb-b54e-c3ea2d956927"

#: A GoodMem id, as a DSPy agent sees it in a tool's argument schema.
#:
#: A ``TypeAliasType`` rather than a bare ``Annotated``: ``dspy.Tool`` reads
#: hints with ``get_type_hints()``, which strips ``Annotated`` metadata but
#: leaves an alias intact, so the pattern reaches the schema the model is
#: shown and ``dspy.Tool`` checks each call against it. To a type checker it
#: is ``str``. The guard that holds for every caller is :func:`require_uuid`.
UUIDStr = TypeAliasType(
    "UUIDStr",
    Annotated[
        str,
        Field(
            pattern=UUID_PATTERN,
            description=f"A GoodMem id: a UUID such as {_EXAMPLE}. Anything else is refused.",
        ),
    ],
)


class GoodMemIdError(ValueError):
    r"""Raised when an id is not a canonical UUID."""


def _shown(value: Any) -> str:
    r"""Returns a short, printable rendering of a refused value."""
    text = repr(value)
    return text if len(text) <= 80 else f"{text[:77]}..."


def require_uuid(value: Any, field: str) -> str:
    r"""Returns ``value`` as a lowercase canonical UUID, or refuses it.

    Args:
        value (Any): The id a caller, a model or configuration supplied. A
            :class:`uuid.UUID` is accepted as well as its string form.
        field (str): The argument's name, used in the error message.

    Returns:
        str: The id, lowercased.

    Raises:
        GoodMemIdError: If ``value`` is anything but a canonical UUID --
            including braces, a ``urn:uuid:`` prefix, missing hyphens or
            surrounding whitespace, all of which :class:`uuid.UUID` accepts.
    """
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str) and _UUID.fullmatch(value):
        return value.lower()
    raise GoodMemIdError(
        f"{field} must be a UUID such as {_EXAMPLE}; got {_shown(value)}. GoodMem ids are UUIDs, "
        "and any other value could be resolved into a different request path, so it is refused "
        "before a request is made."
    )


def require_uuids(values: Iterable[Any], field: str) -> list[str]:
    r"""Applies :func:`require_uuid` to each id, naming the position refused.

    Args:
        values (Iterable[Any]): The ids.
        field (str): The argument's name; an error names ``field[i]``.

    Returns:
        List[str]: The ids, lowercased, in order.
    """
    return [require_uuid(value, f"{field}[{index}]") for index, value in enumerate(values)]
