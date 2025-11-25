from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import pint
from pint.compat import fully_qualified_name, upcast_type_map
from symbolite.ops import translate
from symbolite.impl import libstd

if TYPE_CHECKING:
    from .types import Derivative

T = TypeVar("T", bound=type)


def register_with_pint(cls: T) -> T:
    """Register type with Pint to  to return NotImplemented
    on methods that can be reflected, such as __add__,
    by adding the type to pint's upcast map.
    """
    upcast_type_map[fully_qualified_name(cls)] = cls
    return cls


class EvalUnitError(Exception):
    pass


def try_eval_units(value):
    try:
        return translate(value, libsl=libstd)
    except EvalUnitError:
        return None


def check_equations_units(lhs: Derivative, rhs):
    order = 0
    if (value := lhs.variable.initial) is None:
        # Maybe a derivative has a unit already assigned
        for order, der in lhs.variable.derivatives.items():
            if (value := der.initial) is not None:
                break
        else:
            # No unit assigned. Only check that rhs is consistent.
            try_eval_units(rhs)
            return

    value = try_eval_units(value)
    rhs = try_eval_units(rhs)
    if rhs is None:
        return
    if isinstance(value, pint.Quantity):
        time = value._REGISTRY.s
    elif isinstance(rhs, pint.Quantity):
        time = rhs._REGISTRY.s
    else:
        return

    value / time ** (lhs.order - order) - rhs  # check units
    return


def check_units(var, value):
    lhs = try_eval_units(var)
    rhs = try_eval_units(value)

    if lhs is not None and rhs is not None:
        lhs - rhs  # must have same units
        return


def check_derivative_units(derivative: Derivative, value):
    check_units(derivative, value)
    check_equations_units(derivative, value)
