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
#: shown and ``dspy.Tool`` checks each call against it -- on DSPy 3.x. DSPy
#: 2.5 shows the model the bare name ``UUIDStr`` and checks nothing. To a
#: type checker it is ``str``. The guard that holds for every caller, on
#: every DSPy version, is :func:`require_uuid`.
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
    try:
        text = str.__repr__(value) if issubclass(type(value), str) else repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    if not issubclass(type(text), str):
        text = f"<{type(value).__name__}>"
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

    The id returned is always a plain ``str`` built from the very characters
    the pattern matched. A ``str`` subclass can override ``lower()``,
    ``__str__`` and ``__format__``, and a :class:`uuid.UUID` subclass its
    ``__str__``, so neither is asked for its value after the check: a
    subclass is checked and sent by what it holds, and a :class:`uuid.UUID`
    whose text is not a canonical UUID is refused. Types are read with
    ``type()``, which an object cannot fake the way it can ``__class__``.
    """
    text: Any = value
    if issubclass(type(value), uuid.UUID):
        try:
            text = str(value)
        except Exception:
            text = None
    if issubclass(type(text), str) and _UUID.fullmatch(text):
        return str.lower(text)
    raise GoodMemIdError(
        f"{field} must be a UUID such as {_EXAMPLE}; got {_shown(value)}. GoodMem ids are UUIDs, "
        "and any other value could be resolved into a different request path, so it is refused "
        "before a request is made."
    )


def id_list(values: Any) -> list[Any]:
    r"""Returns the ids a one-or-many argument holds, without checking them.

    A lone ``str`` or :class:`uuid.UUID` is one id; anything else is
    iterated. A ``uuid.UUID`` is not iterable, so without this a single one
    raised ``TypeError`` instead of being used.

    Args:
        values (Any): One id, or an iterable of ids.

    Returns:
        List[Any]: The ids, still to be checked by :func:`require_uuids`.
    """
    if issubclass(type(values), str | uuid.UUID):
        return [values]
    return list(values)


def require_uuids(values: Iterable[Any], field: str) -> list[str]:
    r"""Applies :func:`require_uuid` to each id, naming the position refused.

    Args:
        values (Iterable[Any]): The ids.
        field (str): The argument's name; an error names ``field[i]``.

    Returns:
        List[str]: The ids, lowercased, in order.
    """
    return [require_uuid(value, f"{field}[{index}]") for index, value in enumerate(values)]
