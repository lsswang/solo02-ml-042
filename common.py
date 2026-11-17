"""Shared validation helpers for the alignment and subtitle modules.

The public functions in this project deliberately use plain Python objects so
the same code runs with or without NumPy/SciPy.  All validation happens before
any solver state is built: malformed input raises :class:`InputError` instead of
being coerced, ignored, or reported as an infeasible optimisation problem.
"""

from __future__ import annotations

import math
import numbers
from typing import Any


class InputError(ValueError):
    """Raised when input does not satisfy the documented data contract."""


def require_sequence(value: Any, label: str) -> tuple:
    """Return a tuple copy of a non-string sequence."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (list, tuple)
    ):
        raise InputError(f"{label} must be a list or tuple")
    return tuple(value)


def require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be a mapping")
    return value


def _check_surrogates(text: str, label: str) -> None:
    """Reject lone UTF-16 surrogate code points.

    Python strings normally contain Unicode scalar values, but callers can
    construct strings with lone surrogates (for example by slicing a surrogate
    pair).  Paired surrogates are still a valid encoding of one scalar value,
    so they are accepted and count as one display scalar.
    """

    i = 0
    n = len(text)
    while i < n:
        code = ord(text[i])
        if 0xD800 <= code <= 0xDBFF:
            if i + 1 >= n or not (0xDC00 <= ord(text[i + 1]) <= 0xDFFF):
                raise InputError(f"{label} contains an unpaired surrogate")
            i += 2
            continue
        if 0xDC00 <= code <= 0xDFFF:
            raise InputError(f"{label} contains an unpaired surrogate")
        i += 1


def require_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise InputError(f"{label} must be a string")
    if not allow_empty and value == "":
        raise InputError(f"{label} must be non-empty")
    _check_surrogates(value, label)
    return value


def scalar_len(text: str) -> int:
    """Number of Unicode scalar values in ``text``.

    A surrogate pair written as two Python code points counts as one scalar.
    The caller is expected to have validated the string first.
    """

    count = 0
    i = 0
    n = len(text)
    while i < n:
        code = ord(text[i])
        if (
            0xD800 <= code <= 0xDBFF
            and i + 1 < n
            and 0xDC00 <= ord(text[i + 1]) <= 0xDFFF
        ):
            i += 2
        else:
            i += 1
        count += 1
    return count


def is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def require_real(
    value: Any,
    label: str,
    *,
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
) -> float:
    if not is_real(value):
        raise InputError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{label} must be finite")
    if low is not None:
        if (result < low) if low_inclusive else (result <= low):
            raise InputError(f"{label} is below its allowed range")
    if high is not None:
        if (result > high) if high_inclusive else (result >= high):
            raise InputError(f"{label} is above its allowed range")
    return result


def require_int(
    value: Any,
    label: str,
    *,
    low: int | None = None,
    high: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InputError(f"{label} must be an integer")
    result = int(value)
    if low is not None and result < low:
        raise InputError(f"{label} is below its allowed range")
    if high is not None and result > high:
        raise InputError(f"{label} is above its allowed range")
    return result


NEG_INF = float("-inf")


def logadd(a: float, b: float) -> float:
    """Stable ``log(exp(a) + exp(b))``."""

    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    # At this point a is the larger value, so b-a <= 0.
    return a + math.log1p(math.exp(b - a))
