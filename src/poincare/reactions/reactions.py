from typing import Sequence, Self, Iterator
from dataclasses import dataclass
from collections import defaultdict

from symbolite import Real
from symbolite.core.call import Call, CallInfo
from symbolite.core.value import ValueInfo
from symbolite.abstract import real
from symbolite.ops import substitute

from ..types import System, Variable, Equation, EquationGroup, Initial, initial
from .._node import Node, NodeMapper


class ReactionVariable(Node, Real):
    def __init__(
        self,
        variable: Variable | Initial,
        stoichiometry: float = 1,
    ):
        if not isinstance(variable, Variable):
            variable = Variable(initial=variable)
        self.variable = variable
        self.stoichiometry = stoichiometry

    def __set_name__(self, cls: Node, name: str):
        return self.variable.__set_name__(cls, name)

    @property
    def name(self):
        return self.variable.name

    @property
    def parent(self):
        return self.variable.parent

    def _copy_from(self, parent: Node) -> Self:
        variable = self.variable._copy_from(parent)
        return ReactionVariable(variable, self.stoichiometry)

    def __get__(self, parent, cls):
        if parent is None:
            return self

        reaction_variable = super().__get__(parent, cls)
        return ReactionVariable(
            reaction_variable.variable,
            self.stoichiometry * reaction_variable.stoichiometry,
        )

    def __set__(self, obj, value: Initial | Self):
        if isinstance(value, Initial):
            reaction_variable = initial(default=value)
        else:
            try:
                reaction_variable = ReactionVariable.from_mul(value)
            except TypeError:
                raise TypeError(f"unexpected type {type(value)} for {self.name}")
            else:
                reaction_variable.stoichiometry *= self.stoichiometry

        super().__set__(obj, reaction_variable)

    def __repr__(self):
        return f"{self.stoichiometry} * {self.variable}"

    def __str__(self):
        return str(self.variable)

    def __hash__(self):
        return hash((self.variable, self.stoichiometry))

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.variable, self.stoichiometry) == (
            other.variable,
            other.stoichiometry,
        )

    @classmethod
    def from_mul(cls, expr: Real):
        match expr:
            case Variable() as var:
                return ReactionVariable(var, 1)
            case ReactionVariable(variable=var, stoichiometry=st):
                return ReactionVariable(var, st)

            case Real(
                __symbolite_info__=ValueInfo(
                    value=Call(
                        __symbolite_info__=CallInfo(
                            func=real.mul,
                            args=(
                                int(st2) | float(st2),
                                ReactionVariable(variable=var, stoichiometry=st),
                            )
                            | (
                                ReactionVariable(variable=var, stoichiometry=st),
                                int(st2) | float(st2),
                            ),
                        )
                    )
                )
            ):
                return ReactionVariable(var, st * st2)

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
                return ReactionVariable(var, st)

            case _:
                raise TypeError


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
    ):
        self.reactants = tuple(map(ReactionVariable.from_mul, reactants))
        self.products = tuple(map(ReactionVariable.from_mul, products))
        self.rate_law = rate_law
        self.equations = tuple(self._yield_equations())

    def _copy_from(self, parent: System):
        mapper = NodeMapper(parent)
        return self.__class__(
            reactants=[
                ReactionVariable(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.reactants
            ],
            products=[
                ReactionVariable(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.products
            ],
            rate_law=substitute(self.rate_law, mapper),
        )

    def _yield_equations(self) -> Iterator[Equation]:
        species_stoich: dict[Variable, float] = defaultdict(float)
        for r in self.reactants:
            species_stoich[r.variable] -= r.stoichiometry
        for p in self.products:
            species_stoich[p.variable] += p.stoichiometry

        for s, st in species_stoich.items():
            yield s.derive() << st * self.rate_law


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
        self.reactants = tuple(map(ReactionVariable.from_mul, reactants))
        self.products = tuple(map(ReactionVariable.from_mul, products))
        self.rate = rate
        self.equations = tuple(self._yield_equations())

    @property
    def rate_law(self):
        rate = self.rate
        for r in self.reactants:
            rate *= r.variable**r.stoichiometry
        return rate

    def _copy_from(self, parent: System):
        mapper = NodeMapper(parent)
        return self.__class__(
            reactants=[
                ReactionVariable(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.reactants
            ],
            products=[
                ReactionVariable(substitute(v.variable, mapper), v.stoichiometry)
                for v in self.products
            ],
            rate=substitute(self.rate, mapper),
        )
