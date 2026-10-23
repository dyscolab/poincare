from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any

import pint
from pint.compat import fully_qualified_name, upcast_type_map
from symbolite.impl import libstd
from symbolite.ops import substitute, translate

from ._node import Node, class_namespace

if TYPE_CHECKING:
    from .types import Derivative, System

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


@singledispatch
def strip_node(node: Node, stripper: UnitStripper) -> Node:
    """Create a copy of node with all quantities replaced by their
    magnitude in base units, and all nodes mapped by stripper.

    Node subclasses register their implementation, mirroring their _copy_from.
    """
    raise NotImplementedError(f"cannot strip units from {type(node)}")


class UnitStripper:
    """Mapping-like object for symbolite.substitute that replaces quantities
    by their magnitude in base units, and nodes of old_cls by their unitless copy.
    """

    def __init__(self, old_cls: type[System] | None):
        self.old_cls = old_cls
        self.namespace: dict[str, Any] = {}
        self.classes: dict[type[System], type[System]] = {}

    def get(self, item, default=None):
        from .types import Derivative

        match item:
            case pint.Quantity():
                return item.to_base_units().magnitude
            case pint.Unit():
                return (1 * item).to_base_units().magnitude
            case Derivative(variable=variable, order=order):
                variable = self.get(variable)
                try:
                    # Derivative(variable, order=order) would reset its initial
                    return variable.derivatives[order]
                except KeyError:
                    return Derivative(variable, order=order)
            case Node(parent=None):
                return item
            case Node(parent=type() as parent, name=name) if (
                self.old_cls is not None and issubclass(self.old_cls, parent)
            ):
                # Nodes of old_cls, including those inherited from abstract bases
                try:
                    return self.namespace[name]
                except KeyError:
                    new = self.namespace[name] = strip_node(item, self)
                    return new
            case Node(parent=Node() as parent, name=name):
                new_parent = self.get(parent)
                if item is getattr(parent, "variable", None):
                    # Reactant variables
                    return new_parent.variable
                return getattr(new_parent, name)
            case _:
                return default

    def substitute(self, value):
        return substitute(value, self)

    def strip_class(self, cls: type[System]) -> type[System]:
        try:
            return self.classes[cls]
        except KeyError:
            new = self.classes[cls] = strip_units(cls)
            return new


def strip_units[S: type[System] | System](system: S) -> S:
    """Return a copy of system without units.

    Every quantity is replaced by its magnitude in base units,
    as the Simulator does internally.
    The removed units are stored in `_units`,
    a dict mapping the new system's components to base units,
    which can be used to add them back.
    Components that were plain numbers map to None,
    so they are not turned into dimensionless quantities when adding units back.
    """
    from .types import System

    if isinstance(system, System):
        stripper = UnitStripper(None)
        new_cls = strip_units(type(system))
        new = new_cls(**{k: stripper.substitute(v) for k, v in system._kwargs.items()})
        new._units = _rekey_units(collect_units(system), new)
        return new

    old_cls = system
    namespace = System.__prepare__(old_cls.__name__, (System,))
    annotations = namespace.setdefault("__annotations__", {})
    stripper = UnitStripper(old_cls)
    for name, value in class_namespace(old_cls).items():
        if not isinstance(value, Node):
            continue
        namespace[name] = stripper.get(value)
        if name in old_cls._annotations:
            annotations[name] = old_cls._annotations[name]

    new_cls = type(old_cls.__name__, (System,), namespace)
    new_cls._units = _rekey_units(collect_units(old_cls), new_cls)
    return new_cls


def _rekey_units(
    units: dict[Node | str, pint.Unit | None], system: type[System] | System
) -> dict[Node | str, pint.Unit | None]:
    """Replace each component key by the component with the same path in system."""
    from .helpers import get_from

    return {
        get_from(str(k), system) if isinstance(k, Node) else k: v
        for k, v in units.items()
    }


def collect_units(
    system: type[System] | System,
) -> dict[Node | str, pint.Unit | None]:
    """Base units of each component of system.

    Plain numbers map to None, which is different from quantities whose units
    cancel out, which map to dimensionless. Components without a value are skipped.
    If no Independent is declared, its units are keyed by "time",
    the name of the Simulator's time coordinate.
    """
    from .types import Constant, Derivative, Independent, Parameter, Variable

    values = {}
    components = Independent | Constant | Parameter | Variable | Derivative
    for node in system._yield(components):
        if node.name == "":
            continue
        values[node] = try_eval_units(node)

    registry = next(
        (v._REGISTRY for v in values.values() if isinstance(v, pint.Quantity)),
        ureg,
    )
    out = {}
    for node, value in values.items():
        match value:
            case None:
                continue
            case pint.Quantity():
                out[node] = value.to_base_units().units
            case _:
                out[node] = None

    # The independent's units can be implicit, inferred from the derivatives.
    dimensionality = system._independent_units
    independent = list(system._yield(Independent, recursive=False))
    for key in independent or ["time"]:
        if out.get(key) is not None:
            continue
        elif dimensionality in (None, MISSING):
            out[key] = None
        else:
            out[key] = base_units_from_dimensionality(dimensionality, registry)
    return out


def base_units_from_dimensionality(
    dimensionality: pint.util.UnitsContainer, registry: pint.UnitRegistry
) -> pint.Unit:
    unit = registry.dimensionless
    for dimension, exponent in dimensionality.items():
        reference = next(iter(registry.get_compatible_units(dimension)))
        unit *= (1 * reference).to_base_units().units ** exponent
    return unit
