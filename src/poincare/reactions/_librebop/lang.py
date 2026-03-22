from __future__ import annotations

from typing import Any

from symbolite.core import Unsupported
from symbolite.ops._translate import translate

Block = Unsupported

Assign = Unsupported


def to_bool(value: bool, libsl: Any) -> str:
    return "True" if value else "False"


def to_int(value: int, libsl: Any) -> str:
    return repr(value)


def to_float(value: float, libsl: Any) -> str:
    return repr(value)


def to_tuple(value: tuple[Any, ...], libsl: Any) -> str:
    value = (translate(v, libsl) for v in value)
    return f"({', '.join(map(str, value))}, )"


to_list = Unsupported

to_dict = Unsupported


__all__ = [
    "Block",
    "Assign",
    "to_bool",
    "to_int",
    "to_float",
    "to_tuple",
]
