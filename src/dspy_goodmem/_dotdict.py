"""Vendored copy of ``dspy.dsp.utils.dotdict``.

Vendored because ``dspy.dsp.utils`` is DSPy-internal and not part of its
public API — importing it directly risks breakage on DSPy internal
refactors.  The class itself is a trivial ``dict`` subclass that exposes
items as attributes; ``GoodMemRM`` uses it to match the return shape
expected by ``dspy.Retrieve``.
"""

from __future__ import annotations

import copy


class dotdict(dict):  # noqa: N801
    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            return super().__getattr__(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("__") and key.endswith("__"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __delattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        return dotdict(copy.deepcopy(dict(self), memo))
