import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

try:
    from rebop import Gillespie
    from rebop.gillespie import RNGLike, SeedLike
except ImportError:
    raise ImportError(
        "Stochastic simulations require the rebop library to be installed"
    )
import pint
from symbolite import Real, substitute, translate
from symbolite.ops import yield_named

from ..._node import Node
from ...simulator import Simulator
from ...types import Constant, Equation, Number, Parameter, System
from ..reactions import MassAction, RateLaw, Reactant
from . import _librebop


class RebopSimulator:
    def __init__(
        self,
        model: type[System],
        /,
    ):
        for eq in model._yield(Equation):
            raise TypeError("Non reaction equations can't be simulated with rebop")
        self.model = model
        self._sim = Simulator(model)
        self._sim.compiled = dataclasses.replace(self._sim.compiled, func=None)
        self._variable_map = {
            k: str(k).replace(".", "__") for k in self._sim.compiled.variables
        }

    def _build(self, p: Mapping[Parameter, float], /):
        rebop = Gillespie()
        for r in self.model._yield(RateLaw):
            rebop_rate = self._get_rebop_rate(r, p)
            rebop.add_reaction(
                rate=rebop_rate,
                reactants=list(self._yield_reactants(r.reactants)),
                products=list(self._yield_reactants(r.products)),
            )
        return rebop

    def _yield_reactants(self, reactants: Iterable[Reactant], /):
        for r in reactants:
            if not r.stoichiometry.is_integer():
                raise NotImplementedError("Only integer stoichiometries are allowed")

            name = self._variable_map[r.variable]
            for _ in range(int(r.stoichiometry)):
                yield name

    def solve(
        self,
        values: Mapping = {},
        *,
        upto_t: float,
        n_points: int | None = None,
        rng: RNGLike | SeedLike | None = None,
        sparse: bool = True,
        var_names: Iterable[Reactant] | None = None,
    ):
        if n_points is None:
            n_points = 0

        if var_names is not None:
            var_names = [self._variable_map[v.variable] for v in var_names]

        problem = self._sim.create_problem(values)
        for s in problem.scale:
            if isinstance(s, pint.Quantity | pint.Unit):
                raise TypeError("Stochastic simulation doesn't support units")
        y = {k: int(v) for k, v in zip(self._variable_map.values(), problem.y)}
        p = dict(zip(self._sim.compiled.parameters, problem.p))
        rebop = self._build(p)
        ds = rebop.run(
            y,
            tmax=upto_t,
            nb_steps=n_points,
            rng=rng,
            sparse=sparse,
            var_names=var_names,
        )
        # Replace "__" for "." in output dataset and sort data_vars alphabetically to make output consistent with the rest of poincare
        ds = ds.rename_vars(
            name_dict={new: str(old) for old, new in self._variable_map.items()},
        )  # TODO: inplace = True would be more efficient? xarray errors when trying to set it
        ds = ds[sorted(list(ds.data_vars))]
        return ds

    def _get_rebop_rate(self, r: RateLaw, p: Mapping[Parameter:Number]):
        if isinstance(r, MassAction):
            if isinstance(r.rate, Number):
                return r.rate
            if isinstance(r.rate, Constant):
                return get_by_name(p, r.rate)
            if isinstance(r.rate, Parameter) and isinstance(r.rate.default, Number):
                return p[r.rate]

        # Replace parameters and constant for numeric values or expressions that depend only on Variables
        new_expression = {}
        for named in yield_named(r.rate_law):
            if isinstance(named, Parameter | Constant):
                new_expression[named] = replace_algebraic_expressions(named, p)
        no_algebraics = substitute(r.rate_law, new_expression)

        # Replace Variables for pure Reals of the same name so it will translate to it's name and not it's initial
        reps = {}
        for named in yield_named(no_algebraics):
            if isinstance(named, Node):
                # TODO: is there a legitimate reason for there to be Nodes other than Variable? Or should it throw an error if it finds any?
                reps[named] = Real(self._variable_map.get(named, named.name))
        new_rate = substitute(no_algebraics, reps)
        translated = translate(new_rate, _librebop)
        return getattr(translated, "text", translated)


def replace_algebraic_expressions(
    expr: Real, p: Mapping[Parameter, Number]
) -> Real | Number:
    if isinstance(expr, Constant):
        return get_by_name(p, expr)
    elif isinstance(expr, Parameter):
        if isinstance(expr.default, Real):
            substitutions = {
                named: replace_algebraic_expressions(named, p)
                for named in yield_named(expr.default)
            }
            return substitute(expr.default, substitutions)
        else:
            return p[expr]
    else:
        return expr


def get_by_name(dict: Mapping[Node, Any], item: Node) -> Any:
    for k, v in dict.items():
        if k.name == item.name:
            return v
    raise KeyError("No key matches item's name")
