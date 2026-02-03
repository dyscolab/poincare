from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import singledispatch
from types import ModuleType
from typing import Any, ClassVar, Literal, Self, TypeVar, dataclass_transform, overload
from typing import get_type_hints as get_annotations

import numpy as np
import pandas as pd
import pint
from symbolite import Real
from symbolite import abstract as libabstract
from symbolite.core import Value
from symbolite.ops import substitute, translate

from . import units
from ._node import Node, NodeMapper
from .printing.table import Table

T = TypeVar("T")


units.register_with_pint(Real)


@units.register_with_pint
class Constant(Node, Real):
    def __init__(self, *, default: Initial | None):
        super().__init__()
        self.default = default
        units.check_units(self, default)

    def _copy_from(self, parent: Node):
        return self.__class__(default=substitute(self.default, NodeMapper(parent)))

    def __eq__(self, other: Any):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.default == other.default and super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __set__(self, obj, value: Initial | Constant):
        if isinstance(value, Constant):
            super().__set__(obj, value)
        elif isinstance(value, Initial):
            # Get or create instance with getattr
            constant: Constant = getattr(obj, self.name)
            units.check_units(constant, value)
            constant.default = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")


Number = int | float | complex
Initial = Number | Constant | pint.Quantity
Array1d = np.ndarray[tuple[int], np.dtype[np.floating]]


def _assign_equation_order(
    *,
    variable: Variable,
    order: int,
):
    if variable.equation_order is not None and variable.equation_order != order:
        raise ValueError(
            f"already assigned an equation to order {variable.equation_order}"
        )

    for der_order, der in variable.derivatives.items():
        if der_order >= order and der.initial is not None:
            raise ValueError(
                f"already assigned an initial to a higher order {der.order}"
            )

    variable.equation_order = order


@units.register_with_pint
class Parameter(Node, Real):
    equation_order: int = 0

    def __init__(self, *, default: Initial | Value | None):
        super().__init__()
        self.default = default
        units.check_units(self, default)

    def _copy_from(self, parent: Node):
        return self.__class__(default=substitute(self.default, NodeMapper(parent)))

    def __set__(self, obj, value: Initial | Value):
        if isinstance(value, Parameter):
            super().__set__(obj, value)
        elif isinstance(value, Initial | Value):
            # Get or create instance with getattr
            # Update initial value
            variable: Self = getattr(obj, self.name)
            units.check_units(variable, value)
            variable.default = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Any):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.default == other.default and super().__eq__(other)


@translate.register
def translate_constant_parameter(obj: Constant | Parameter, libsl: ModuleType) -> Any:
    if libsl is libabstract:
        return obj
    elif obj.default is None:
        raise units.EvalUnitError
    else:
        return translate(obj.default, libsl)


@units.register_with_pint
class Variable(Node, Real):
    initial: Initial | None
    derivatives: dict[int, Derivative]
    equation_order: int | None = None

    def __init__(self, *, initial: Initial | None):
        super().__init__()
        self.initial = initial
        self.derivatives = {}
        units.check_units(self, initial)

    def __getnewargs__(self):
        return (self.initial, self.derivatives, self.equation_order)

    def derive(self, *, initial: Initial | None = None) -> Derivative:
        return Derivative(self, initial=initial, order=1)

    def _copy_from(self, parent: Node):
        mapper = NodeMapper(parent)
        copy = self.__class__(initial=substitute(self.initial, mapper))
        for order, der in self.derivatives.items():
            copy.derivatives[order] = Derivative(
                copy, initial=substitute(der.initial, mapper), order=order
            )
        copy.equation_order = self.equation_order
        return copy

    def __set__(self, obj, value: Variable | Initial):
        if not isinstance(value, Initial | Variable):
            raise TypeError("unexpected type")

        try:
            variable = obj.__dict__[self.name]
        except KeyError:
            # Variable not yet created
            if isinstance(value, Variable):
                # Replace descriptor by adding it to obj.__dict__
                # Update derivatives initials
                super().__set__(obj, value)
                if (order := self.equation_order) is not None:
                    _assign_equation_order(variable=value, order=order)
                for order, self_der in self.derivatives.items():
                    if self_der.name == "":
                        continue  # derivative taken but not assigned
                    try:
                        value_der = value.derivatives[order]
                    except KeyError:
                        raise TypeError(
                            "must explicitly define higher order derivatives in the external Variable"
                        )
                    else:
                        obj.__dict__[self_der.name] = value_der
            elif isinstance(value, Initial):
                # Create internal variable and update its initial
                variable: Variable = getattr(obj, self.name)
                units.check_units(variable, value)
                variable.initial = value
        else:
            # Variable already assigned (by a Derivative)
            if isinstance(value, Variable):
                if value is not variable:
                    raise TypeError("wrongly assigned variable or derivative")
            elif isinstance(value, Initial):
                if variable.parent is not obj:
                    raise TypeError("cannot assign initial to an external variable")
                variable.initial = value

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Any):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.initial == other.initial and super().__eq__(other)


@units.register_with_pint
class Derivative(Node, Real):
    def __new__(
        cls,
        variable: Variable,
        *,
        initial: Initial | None = None,
        order: int,
    ):
        try:
            der = variable.derivatives[order]
        except KeyError:
            return super().__new__(cls)

        if initial is None:
            return der
        else:
            raise ValueError(
                f"already assigned an initial value to order {order} of variable {variable}"
            )

    def __init__(
        self,
        variable: Variable,
        *,
        initial: Initial | None = None,
        order: int,
    ):
        super().__init__()
        if self.name != "":
            # Existing derivative
            return
        self.variable = variable
        self.order = order
        self.initial = initial
        units.check_derivative_units(self, initial)

    def __set_name__(self, cls: Node, name: str):
        super().__set_name__(cls, name)
        self.variable.derivatives[self.order] = self

    def __getnewargs_ex__(self):
        args = (self.variable,)
        kwargs = {
            "initial": self.initial,
            "order": self.order,
        }
        return args, kwargs

    def _copy_from(self, parent: Node) -> Self:
        variable: Variable = getattr(parent, self.variable.name)
        return variable.derivatives[self.order]

    def __set__(self, obj, value: Initial | Derivative):
        if not isinstance(value, Initial | Derivative):
            raise TypeError("unexpected type")

        # 3 cases to check:
        # - 1. variable and derivative are internal (assigned an Initial or using default)
        # - 2. variable and derivative are external (assigned Variable and Derivative)
        # - 3. variable and internal are mixed internal/external -> error
        # The order of assignment matters.

        try:
            derivative: Derivative = obj.__dict__[self.name]
        except KeyError:
            # derivative has not been assigned -> variable has not been assigned
            if isinstance(value, Derivative):
                # it refers to an outside Variable, let's assign that first
                setattr(obj, self.variable.name, value.variable)
            elif isinstance(value, Initial):
                # this creates an internal variable, which will be an error
                # if it is going to be assigned to an outside variable
                derivative: Derivative = getattr(obj, self.name)
                units.check_derivative_units(derivative, value)
                derivative.initial = value
        else:
            # derivative has been assigned -> variable has been assigned
            if isinstance(value, Initial):
                # existing derivative must be internal to replace its initial value
                if derivative.parent is not obj:
                    raise TypeError(
                        "derivative corresponds to an external variable, cannot change its initial condition here."
                    )
                else:
                    units.check_derivative_units(derivative, value)
                    derivative.initial = value
            elif isinstance(value, Derivative) and value is not derivative:
                raise TypeError("assigned wrong derivative")

    def derive(self, *, initial: Initial | None = None) -> Derivative:
        if self.name == "":
            raise NameError("must assign this derivative to a variable first")
        return Derivative(self.variable, initial=initial, order=self.order + 1)

    def __lshift__(self, other) -> Equation:
        return Equation(self, other)

    def __hash__(self) -> int:
        return hash((self.variable, self.order))

    def __eq__(self, other: Any) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable

    def __repr__(self):
        return f"D({self.variable.name}, {self.order})"

    def __str__(self):
        s = super().__str__()
        if s == "":
            return repr(self)
        else:
            return s


@translate.register
def translate_variable_derivative(obj: Variable | Derivative, libsl: ModuleType) -> Any:
    if libsl is libabstract:
        return obj
    elif obj.initial is None:
        raise units.EvalUnitError
    else:
        return translate(obj.initial, libsl)


@dataclass
class Equation(Node):
    lhs: Derivative
    rhs: Initial | Value

    def __post_init__(self):
        units.check_equations_units(self.lhs, self.rhs)

    def __set_name__(self, cls, name):
        super().__set_name__(cls, name)
        _assign_equation_order(variable=self.lhs.variable, order=self.lhs.order)

    def _copy_from(self, parent: Node):
        mapper = NodeMapper(parent)
        variable = mapper.get(self.lhs.variable)
        lhs = Derivative(variable, order=self.lhs.order)
        rhs = substitute(self.rhs, mapper)
        return self.__class__(lhs, rhs)

    def __repr__(self):
        return f"Equation({self.lhs} << {self.rhs})"

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class EquationGroup(Node):
    equations: Sequence[Equation]

    def __init__(self, *equations: Equation):
        self.equations = equations

    def __set_name__(self, cls: Node, name: str):
        for eq in getattr(self, "equations", []):
            _assign_equation_order(variable=eq.lhs.variable, order=eq.lhs.order)
        super().__set_name__(cls, name)

    def _copy_from(self, parent: Node):
        return self.__class__(*(eq._copy_from(parent) for eq in self.equations))

    def __hash__(self) -> int:
        return super().__hash__()


@overload
def assign(
    *,
    default: Initial | Value | None = None,
    constant: Literal[False] = False,
    init: bool = True,
) -> Parameter: ...


@overload
def assign(
    *,
    default: Initial | None = None,
    constant: Literal[True],
    init: bool = True,
) -> Constant: ...


def assign(*, default=None, constant: bool = False, init: bool = True):
    if constant:
        return Constant(default=default)
    else:
        return Parameter(default=default)


def initial(*, default: Initial | None = None, init: bool = True) -> Variable:
    return Variable(initial=default)


@units.register_with_pint
class Independent(Node, Real):
    def __init__(self, *, default: Initial | Value | None = 0):
        super().__init__()
        self.default = default
        units.check_units(self, default)

    def _copy_from(self, parent: Node):
        return self.__class__(default=substitute(self.default, NodeMapper(parent)))

    def __get__(self, obj: System | None, cls):
        if obj is None or obj.parent is None:
            return self

        independent = set(obj.parent._yield(self.__class__, recursive=False))
        match len(independent):
            case 0:
                raise NotImplementedError("set independent on parent")
            case 1:
                return independent.pop()
            case _:
                raise NotADirectoryError("more than one independent variable")

    def __set__(self, obj, value: Initial | Value):
        if isinstance(value, Independent):
            super().__set__(obj, value)
        elif isinstance(value, Initial | Value):
            # Get or create instance with getattr
            # Update initial value
            variable: Self = getattr(obj, self.name)
            units.check_units(variable, value)
            variable.default = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Any):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.default == other.default and super().__eq__(other)


@translate.register
def translate_independent(obj: Independent, libsl: ModuleType) -> Any:
    if libsl is libabstract:
        return obj
    elif obj.default is None:
        raise units.EvalUnitError
    else:
        return translate(obj.default, libsl)


class OwnedNamerDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise NameError(f"duplicate assignment to the same name: {key}")
        if isinstance(value, Node):
            value.__set_name__(None, key)
        return super().__setitem__(key, value)


class EagerNamer(type):
    _abstract: bool

    @classmethod
    def __prepare__(
        cls, name, bases: tuple[type, ...], *, abstract: bool = False, **kwargs
    ):
        def is_abstract(x):
            return isinstance(x, cls) and not x._abstract

        if not abstract:
            # Prevent subclassing System subclassses
            for base in filter(is_abstract, bases):
                if any(map(is_abstract, base.__bases__)):
                    raise TypeError(f"cannot subclass {base.__name__}")

        return OwnedNamerDict()

    def __init__(self, name, bases, namespace, *, abstract: bool = False, **kwargs):
        super().__init__(name, bases, namespace)
        self._abstract = abstract

    def __str__(self):
        return _as_table(self, max_width=None).__str__()

    def __repr__(self):
        return _as_table(self, max_width=80 - 17).__repr__()

    def _repr_html_(self):
        return _as_table(self, max_width=80 - 17)._repr_html_()

    @property
    def variables(self):
        return pd.Series({str(x): x.initial for x in self._yield(Variable)})

    @property
    def parameters(self):
        return pd.Series({str(x): x.default for x in self._yield(Parameter | Constant)})


def _as_table(self: type[System], *, max_width: int | None) -> Table:
    types = dict(
        variables=list(self._yield(Variable)),
        parameters=list(self._yield(Parameter | Constant)),
        equations=list(self._yield(Equation | EquationGroup)),
    )

    return Table(
        table=[
            [
                name,
                str(len(values)),
                _sequence_to_str(values, sep=", ", max_size=max_width),
            ]
            for name, values in types.items()
        ],
        headers=["type", "total", "names"],
        title=self.__name__,
    )


def _sequence_to_str(seq: Iterator, /, *, sep: str, max_size: int | None) -> str:
    if max_size is None:
        return sep.join(map(str, seq))

    count = 0
    tokens = []
    for token in map(str, seq):
        count += len(token) + len(sep)
        if count >= max_size:
            tokens.append("...")
            break
        tokens.append(token)
    return sep.join(tokens)


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(
        initial,
        assign,
    ),
)
@singledispatch
def get_default(node: Node) -> Node | Initial | None:
    return node


@get_default.register
def get_default_variable(node: Variable) -> Initial | None:
    return node.initial


@get_default.register
def get_default_parameter(node: Parameter | Constant) -> Initial | None:
    return node.default


class System(Node, metaclass=EagerNamer):
    parent: System
    _kwargs: dict
    _required: ClassVar[set[str]]
    _annotations: ClassVar[dict[str, type[Variable | Derivative | System]]]

    def __init_subclass__(cls, **kwargs) -> None:
        cls._annotations = get_annotations(cls)

        for k in ("_annotations", "_required", "_kwargs", "name", "parent"):
            del cls._annotations[k]

        # Check mismatched types and compute required
        parameters = []
        cls._required = set()
        mismatched_types: list[tuple[str, type, type]] = []
        for k, annotation in cls._annotations.items():
            v = getattr(cls, k)
            if not isinstance(v, annotation):
                mismatched_types.append((k, annotation, type(v)))

            default = get_default(v)

            if default is None:
                cls._required.add(k)
                default = inspect.Parameter.empty

            parameters.append(
                inspect.Parameter(
                    name=k,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )

        if len(mismatched_types) > 0:
            raise TypeError(
                "\n".join(
                    [f"{k} expected {ann} got {t}" for k, ann, t in mismatched_types]
                )
            )

        cls.__signature__ = inspect.Signature(parameters)

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if isinstance(value, Value) and not isinstance(value, Node):
            # Value is not a descriptor that knows how to recreate itself in the new instance,
            # but might have Nodes inside that need to be recreated.
            return substitute(value, NodeMapper(self))
        else:
            return value

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("positional parameters are not allowed.")

        missing = self._required - kwargs.keys()
        if len(missing) > 0:
            raise TypeError("missing arguments:", missing)

        unexpected = kwargs.keys() - self._annotations.keys()
        if len(unexpected) > 0:
            raise TypeError("unexpected arguments:", unexpected)

        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _copy_from(self, parent: Node):
        # Create a new instance by replacing previous arguments,
        # which were saved in self._kwargs,
        # with the ones from the corresponding instance
        mapper = NodeMapper(parent)
        kwargs = {k: substitute(v, mapper) for k, v in self._kwargs.items()}
        return self.__class__(**kwargs)

    def __eq__(self, other: Self):
        """Check equality between Systems.

        Two Systems are equal if they:
        - are instances of the same class
        - have been instanced with the same arguments
        """
        if other.__class__ is not self.__class__:
            return NotImplemented

        return all(getattr(self, k) == getattr(other, k) for k in self._annotations)

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = ",".join(f"{k}={v}" for k, v in self._kwargs.items())
        return f"{name}({kwargs})"
