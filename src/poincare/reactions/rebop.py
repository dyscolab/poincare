import dataclasses
from collections.abc import Iterable, Mapping

from rebop import Gillespie
from rebop.gillespie import RNGLike, SeedLike

from ..simulator import Simulator
from ..types import Parameter, System
from .reactions import MassAction, RateLaw, Reactant


class RebopSimulator:
    def __init__(
        self,
        model: type[System],
        /,
    ):
        if not all(isinstance(r, MassAction) for r in model._yield(RateLaw)):
            # TODO: check non-reaction equations?
            raise NotImplementedError("only MassAction reactions are implemented")

        self.model = model
        self._sim = Simulator(model)
        self._sim.compiled = dataclasses.replace(self._sim.compiled, func=None)
        self._variable_map = {
            k: str(k).replace(".", "__") for k in self._sim.compiled.variables
        }

    def _build(self, parameters: Mapping[Parameter, float], /):
        rebop = Gillespie()
        for r in self.model._yield(RateLaw):
            if not isinstance(r, MassAction):
                raise NotImplementedError("only MassAction reactions are implemented")
            # TODO: check if rate is number? What about units?
            rebop_rate = parameters[r.rate] if isinstance(r.rate, Parameter) else r.rate
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
        ds = ds[sorted(list(ds.data_vars))]
        return ds
