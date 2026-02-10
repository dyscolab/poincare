from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatch
from types import ModuleType
from typing import Any, Iterator, Self, Sequence

import pint
from symbolite import Real
from symbolite.abstract import real
from symbolite.core.call import Call, CallInfo
from symbolite.core.value import ValueInfo
from symbolite.ops import substitute, translate

from .._node import Node, NodeMapper, _ClassInfo
from .._utils import class_and_instance_method
from ..types import (
    Equation,
    EquationGroup,
    Initial,
    Parameter,
    System,
    Variable,
    get_default,
)


@singledispatch
def compensate_volume(
    v: Variable, rhs: Real | Initial, reaction_is_concentration: bool
) -> Real | Initial:
    return rhs


@singledispatch
def make_concentration(v: Variable) -> Real:
    return v


def as_real(value) -> Real:
    if isinstance(value, Real):
        return value
    else:
        return Real(value)


class Reactant(Node, Real):
    def __init__(
        self,
        variable: Variable | Initial,
        stoichiometry: float = 1,
    ):
        super().__init__()
        if not isinstance(variable, Variable):
            variable = Variable(initial=variable)
        self.variable = variable  # TODO: define interface for accessing variable
        self.stoichiometry = stoichiometry

    def __set_name__(self, cls: Node, name: str):
        super().__set_name__(cls, name)
        self.variable.__set_name__(self, name)

    def _copy_from(self, parent: Node) -> Self:
        variable = self.variable._copy_from(parent)
        return Reactant(variable, self.stoichiometry)

    def __set__(self, obj, value: Initial | Self):
        if isinstance(value, Initial):
            reactant = reaction_initial(default=value)
        else:
            try:
                reactant = Reactant.from_mul(value)
            except TypeError:
                raise TypeError(f"unexpected type {type(value)} for {self.name}")
            else:
                reactant.stoichiometry *= self.stoichiometry
        self.variable.__set__(reactant, reactant.variable)
        super().__set__(obj, reactant)

    def __hash__(self):
        return hash((self.variable, self.stoichiometry))

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.variable, self.stoichiometry) == (
            other.variable,
            other.stoichiometry,
        )

    @class_and_instance_method
    def _yield[T](
        self,
        type: type[T],
        /,
        *,
        exclude: _ClassInfo = (),
        recursive: bool = True,
    ) -> Iterator[T]:
        try:
            if issubclass(type, Variable) and not isinstance(self.variable, exclude):
                yield self.variable
        except TypeError:
            pass
        super()._yield(type, exclude=exclude, recursive=recursive)

    @classmethod
    def from_mul(cls, expr: Real, parent: Node | None = None):
        match expr:
            case Variable() as var:
                new = Reactant(var, 1)
                if parent is not None:
                    new.__class__.__base__.__set_name__(new, parent, var.name)
                return new
            case Reactant() as rvar:
                return rvar

            case Real(
                __symbolite_info__=ValueInfo(
                    value=Call(
                        __symbolite_info__=CallInfo(
                            func=real.mul,
                            args=(
                                int(st2) | float(st2),
                                Reactant() as rvar,
                            )
                            | (
                                Reactant() as rvar,
                                int(st2) | float(st2),
                            ),
                        )
                    )
                )
            ):
                rvar.stoichiometry *= st2
                return rvar

            case Real(
                __symbolite_info__=ValueInfo(
                    value=Call(
                        __symbolite_info__=CallInfo(
                            func=real.mul,
                            args=(int(st) | float(st), Variable() as var),
                        )
                        | (
                            Variable() as var,
                            int(st) | float(st),
                        ),
                    )
                )
            ):
                new = Reactant(var, st)
                if parent is not None:
                    new.__class__.__base__.__set_name__(new, parent, var.name)
                return new

            case _:
                raise TypeError


def reaction_initial(
    *, default: Initial | None = None, stoichiometry: int = 1, init: bool = True
) -> Reactant:
    """Creates an ReactionVariable with a default initial condition."""
    return Reactant(variable=default, stoichiometry=stoichiometry)


@translate.register
def translate_reaction_variable(obj: Reactant, libsl: ModuleType) -> Any:
    return translate(obj.variable, libsl)


@get_default.register
def get_default_reactant(node: Reactant) -> Initial | None:
    return node.variable.initial


@dataclass
class RateLaw(EquationGroup):
    """A RateLaw reaction contains a set of equations transforming
    reactants into products with a given rate law.

    Given:
        - `x, y: Variable`
        - `k: float`

    Then:
        `Reaction(reactants=[2 * x], products=[y], rate_law=k)`

    will contain two equations:
        - `dx/dt = -2 * k`
        - `dy/dt = +k`
    """

    def __init__(
        self,
        *,
        reactants: Sequence[Real],
        products: Sequence[Real],
        rate_law: float | Real,
        concentration: bool = True,
    ):
        self.reactants = tuple(
            map(lambda x: Reactant.from_mul(x, parent=self), reactants)
        )
        self.products = tuple(
            map(lambda x: Reactant.from_mul(x, parent=self), products)
        )
        self.rate_law = (
            Parameter(default=rate_law)
            if isinstance(rate_law, (pint.Quantity, pint.Unit))
            else rate_law
        )  # Symbolite can't compile equations if they have explicit units, so if it has units rate_law must be extracted as a Parameter
        self.concentration = concentration

    def _copy_from(self, parent: System):
        mapper = NodeMapper(parent)
        return self.__class__(
            reactants=[
                Reactant(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.reactants
            ],
            products=[
                Reactant(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.products
            ],
            rate_law=substitute(self.rate_law, mapper),
        )
        # return self.__class__(
        #     reactants=[substitute(v, mapper) for v in self.reactants],
        #     products=[substitute(v, mapper) for v in self.products],
        #     rate_law=substitute(self.rate_law, mapper),
        # )

    def _yield_equations(self) -> Iterator[Equation]:
        species_stoich: dict[Variable, float] = defaultdict(float)
        for r in self.reactants:
            species_stoich[r.variable] -= r.stoichiometry
        for p in self.products:
            species_stoich[p.variable] += p.stoichiometry

        for s, st in species_stoich.items():
            yield (
                s.derive()
                << compensate_volume(
                    s, st * self.rate_law, reaction_is_concentration=self.concentration
                )
            )

    def __set_name__(self, cls: Node, name: str):
        if cls is not None:
            if (
                isinstance(self.rate_law, Parameter)
                and getattr(self.rate_law, "name", "") == ""
            ):
                try:
                    if issubclass(cls, System):
                        setattr(cls, f"_{name}_rate_law", self.rate_law)
                except TypeError:
                    pass
                self.rate_law.__set_name__(cls=cls, name=f"_{name}_rate_law")
            self.equations = tuple(self._yield_equations())
        super().__set_name__(cls, name)


class AbsoluteRateLaw(RateLaw):
    def __init__(
        self,
        *,
        reactants: Sequence[Real],
        products: Sequence[Real],
        rate_law: float | Real,
    ):
        super().__init__(
            reactants=reactants,
            products=products,
            rate_law=rate_law,
            concentration=False,
        )


class MassAction(RateLaw):
    """A MassAction reaction contains a set of equations transforming
    reactants into products with a given rate.
    The reaction's rate law is the product of the rate
    and the reactants raised to their stoichiometric coefficient.

    Given:
        - `x, y: Variable`
        - `k: float`

    Then:
        `Reaction(reactants=[2 * x], products=[y], rate_law=k)`

    will contain two equations:
        - `dx/dt = -2 * k * x**2`
        - `dy/dt = +k * x**2`
    """

    def __init__(
        self,
        *,
        reactants: Sequence[Real],
        products: Sequence[Real],
        rate: float | Real,
    ):
        self.reactants = tuple(
            map(lambda x: Reactant.from_mul(x, parent=self), reactants)
        )
        self.products = tuple(
            map(lambda x: Reactant.from_mul(x, parent=self), products)
        )
        self.rate = rate
        self.concentration = True

    @property
    def rate_law(self):
        rate = self.rate
        for r in self.reactants:
            rate *= make_concentration(r.variable) ** r.stoichiometry
        return (
            Parameter(default=rate)
            if isinstance(rate, (pint.Quantity, pint.Unit))
            else rate
        )

    def _copy_from(self, parent: System):
        mapper = NodeMapper(parent)
        return self.__class__(
            reactants=[
                Reactant(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.reactants
            ],
            products=[
                Reactant(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.products
            ],
            rate=substitute(self.rate, mapper),
        )
