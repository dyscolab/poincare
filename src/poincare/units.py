from __future__ import annotations

from typing import TYPE_CHECKING

import pint
from pint.compat import fully_qualified_name, upcast_type_map
from symbolite.impl import libstd
from symbolite.ops import translate

if TYPE_CHECKING:
    from .types import Derivative

ureg = pint.get_application_registry()


class _Missing:
    """Sentinel type to distinguish that the implicit dimensionality has no units (None)
    from implicit dimensionality not defined (MISSING).
    """

    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


def register_with_pint[T](cls: T) -> T:
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


def equation_implicit_dimensionality(lhs: Derivative, rhs) -> pint.util.UnitsContainer:
    """Infer implicit Independent diemensionality from an equation. Resturns None if there is a value
    but no units and MISSING if there is no value (the inital of some Variable is not defined)"""
    order = 0
    if (value := lhs.variable.initial) is None:
        # Maybe a derivative has a unit already assigned
        for order, der in lhs.variable.derivatives.items():
            if (value := der.initial) is not None:
                break
        else:
            # No unit assigned. Only check that rhs is consistent.
            try_eval_units(rhs)
            return MISSING

    value = try_eval_units(value)
    rhs = try_eval_units(rhs)
    if rhs is None:
        return MISSING
    if not (isinstance(value, pint.Quantity) or isinstance(rhs, pint.Quantity)):
        return

    return (
        getattr(value, "dimensionality", ureg.dimensionless.dimensionality)
        / getattr(rhs, "dimensionality", ureg.dimensionless.dimensionality)
    ) ** (1 / (lhs.order - order))  # check units


def check_units(var, value):
    lhs = try_eval_units(var)
    rhs = try_eval_units(value)

    if lhs is not None and rhs is not None:
        _ = lhs - rhs  # must have same units
        return


def derivative_implicit_dimensionality(
    derivative: Derivative, value
) -> pint.util.UnitsContainer:
    check_units(derivative, value)
    return equation_implicit_dimensionality(derivative, value)
