import dataclasses
from collections.abc import Iterable, Mapping

from rebop import Gillespie
from rebop.gillespie import RNGLike, SeedLike
from symbolite import Real, substitute, translate
from symbolite.ops import yield_named

from .._node import Node
from ..simulator import Simulator
from ..types import (
    Constant,
    Number,
    Parameter,
    System,
)
from . import _librebop
from .reactions import MassAction, RateLaw, Reactant


class RebopSimulator:
    def __init__(
        self,
        model: type[System],
        /,
    ):
        # TODO: check non-reaction equations?

        # if not all(isinstance(r, MassAction) for r in model._yield(RateLaw)):
        #     raise NotImplementedError("only MassAction reactions are implemented")

        self.model = model
        self._sim = Simulator(model)
        self._sim.compiled = dataclasses.replace(self._sim.compiled, func=None)
        self._variable_map = {
            k: str(k).replace(".", "__") for k in self._sim.compiled.variables
        }

    def _build(self, parameters: Mapping[Parameter, float], /):
        rebop = Gillespie()
        for r in self.model._yield(RateLaw):
            # if not isinstance(r, MassAction):
            #     raise NotImplementedError("only MassAction reactions are implemented")
            # TODO: check if rate is number? What about units?
            # rebop_rate = parameters[r.rate] if isinstance(r.rate, Parameter) else r.rate
            rebop_rate = self._get_rebop_rate(r)
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

    def _get_rebop_rate(self, r: RateLaw):
        if isinstance(r, MassAction):
            if isinstance(r.rate, Number):
                return r.rate
            if isinstance(r.rate, Parameter | Constant) and isinstance(
                r.rate.default, Number
            ):
                return r.rate.default
        reps = {}
        for named in yield_named(r.rate_law):
            if isinstance(named, Node):
                reps[named] = Real(self._variable_map.get(named, named.name))
            new_expr = substitute(r.rate_law, reps)
        print(translate(new_expr, _librebop).text)
        return translate(new_expr, _librebop).text
