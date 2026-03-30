"""
symbolite.impl.libpythoncode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities to build Python source snippets from Symbolite expressions.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from symbolite.core.function import Operator
from symbolite.core.symbolite_object import get_symbolite_info

from ._rebop_names import get_rebop_name, get_rebop_precedence


@dataclass(frozen=True)
class RebopExpr:
    """Represents a snippet of Python code plus its precedence."""

    text: str
    precedence: int = 100

    def __str__(self) -> str:
        return self.text


def _coerce(value: Any) -> RebopExpr:
    if isinstance(value, RebopExpr):
        return value
    if isinstance(value, str):
        return RebopExpr(value)
    if isinstance(value, bool):
        return RebopExpr("True" if value else "False")
    if isinstance(value, (int, float, complex)):
        return RebopExpr(repr(value))
    return RebopExpr(str(value))


def _maybe_parenthesize(expr: RebopExpr, precedence: int, *, right: bool) -> str:
    if expr.precedence < precedence:
        return f"({expr.text})"
    if right and expr.precedence == precedence:
        return f"({expr.text})"
    return expr.text


def make_operator(fmt: str, precedence: int, arity: int) -> Any:
    def _operator(*args: Any) -> RebopExpr:
        coerced = tuple(_coerce(arg) for arg in args)
        if arity == 1:
            (value,) = coerced
            formatted = fmt.format(
                _maybe_parenthesize(
                    value,
                    precedence,
                    right=False,
                )
            )
        else:
            leading = coerced[0]
            trailing = coerced[1:]
            formatted_args = [
                _maybe_parenthesize(
                    leading,
                    precedence,
                    right=False,
                ),
                *(
                    _maybe_parenthesize(
                        arg,
                        precedence,
                        right=index == len(trailing) - 1,
                    )
                    for index, arg in enumerate(trailing)
                ),
            ]
            formatted = fmt.format(*formatted_args)
        return RebopExpr(formatted, precedence)

    return _operator


def as_operator(obj: Operator[Any]) -> Any:
    info = get_symbolite_info(obj)
    return make_operator(
        get_rebop_name(obj),
        get_rebop_precedence(obj),
        info.arity,
    )
