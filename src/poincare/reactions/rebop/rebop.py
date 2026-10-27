import dataclasses
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Self

try:
    from rebop import Gillespie
    from rebop.gillespie import RNGLike, SeedLike
except ImportError:
    raise ImportError(
        "Stochastic simulations require the rebop library to be installed"
    )
import numpy as np
import pint
import pint_xarray
import xarray as xr
from symbolite import Real, substitute, translate
from symbolite.impl import libstd
from symbolite.ops import yield_named

from ..._node import Node
from ...compile import depends_on_at_least_one_variable_or_time
from ...simulator import Components, Simulator
from ...solvers import Solution, _transform
from ...types import Constant, Equation, Initial, Number, Parameter, System
from ..reactions import MassAction, RateLaw, Reactant
from . import _librebop

pint.get_application_registry().force_ndarray_like = False
# When pint-xarray is imported it sets it to true, it can break poincare compilation


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

    def with_values(
        self, values: Mapping[Components, Initial], /, *, append: bool = True
    ) -> Self:
        rsim = self.__class__.__new__(self.__class__)
        rsim.model = self.model
        rsim._sim = self._sim.with_values(values, append=append)
        rsim._variable_map = self._variable_map
        return rsim
    
    def with_transform(
        self,
        transform: Sequence[Real] | Mapping[Hashable, Real] | None = None,
        /,
        *,
        append: bool = False,    ) -> Self:
        rsim = self.__class__.__new__(self.__class__)
        rsim.model = self.model
        rsim._sim = self._sim.with_transform(transform, append=append)
        rsim._variable_map = self._variable_map
        return rsim

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
        *,
        upto_t: float | pint.Quantity,
        n_points: int | None = None,
        rng: RNGLike | SeedLike | None = None,
        n_runs: int = 1,
        sparse: bool = True,
        var_names: Iterable[Reactant] | None = None,
    ):
        """Run a stochastic simulation with rebop.

        With ``n_runs > 1``, the simulation is repeated ``n_runs`` times with
        independent random streams drawn from ``rng`` and the average is returned.
        Averaging requires ``n_points`` so that every run shares the same time grid.
        """
        upto_t_dim = getattr(upto_t, "dimensionality", None)
        user_units = getattr(upto_t, "units", None)
        if upto_t_dim != self.model._independent_units:
            raise pint.PintError(
                f"save at dimensionality {upto_t_dim} doesn't match (explicit or implicit) dimensionality of Independent {self.model._independent_units}"
            )

        if isinstance(upto_t, pint.Quantity):
            base_units = upto_t.to_base_units().units
            tmax = upto_t.to_base_units().magnitude
        else:
            base_units = None
            tmax = upto_t

        if n_points is None:
            n_points = 0

        if n_runs < 1:
            raise ValueError(f"n_runs must be a positive integer, got {n_runs}.")
        if n_runs > 1 and n_points == 0:
            raise ValueError(
                "n_points is required when n_runs > 1, so that all runs share the same time points."
            )

        if var_names is not None:
            var_names = [self._variable_map[v.variable] for v in var_names]

        problem = self._sim.create_problem()
        # for s in problem.scale:
        #     if isinstance(s, pint.Quantity | pint.Unit):
        #         raise TypeError(
        #             "Stochastic simulation doesn't support units in values, only in upto_t"
        #         )
        y = {k: int(v) for k, v in zip(self._variable_map.values(), problem.y)}
        p = dict(zip(self._sim.compiled.parameters, problem.p))
        rebop = self._build(p)
        # A single generator gives each run an independent random stream, even when
        # rng is an integer seed.
        rng = np.random.default_rng(rng)
        init = y
        ys = []
        for _ in range(n_runs):
            ds = rebop.run(
                init,
                tmax=tmax,
                nb_steps=n_points,
                rng=rng,
                sparse=sparse,
                var_names=var_names,
            )
            ds = ds[sorted(ds.data_vars)] # Sort data Variables to make order consistent with  the rest of Poincare

            # Apply transform
            solved_y = ds.to_dataarray().values
            t = ds["time"].values
            sol = Solution(t, solved_y)
            solution = _transform(problem=problem, solution= sol)
            ys.append(solution.y)
        y = ys[0] if n_runs == 1 else np.mean(ys, axis=0)

        ds = xr.Dataset(
            {
                k: xr.DataArray(
                    data=x * s.magnitude, dims="time", coords={"time": t}
                ).pint.quantify(
                    s.units, pint_xarray.setup_registry(s.units._REGISTRY)
                )
                if isinstance(s, pint.Quantity)
                else xr.DataArray(data=x * s, dims="time", coords={"time": t})
                for k, s, x in zip(self._sim.transform.output.keys(), problem.scale, y.T)
            }
        )

        for s in problem.scale:
            if isinstance(s, pint.Quantity):
                s.units._REGISTRY.force_ndarray_like = False
        if user_units is not None:
            pint_xarray.setup_registry(user_units._REGISTRY)
            ds = ds.pint.quantify({"time": base_units})
            ds = ds.pint.to({"time": user_units})
            user_units._REGISTRY.force_ndarray_like = False
        return ds

    def _get_rebop_rate(self, r: RateLaw, p: Mapping[Parameter, Number]):
        if isinstance(r, MassAction):
            if isinstance(r.rate, Number):
                return r.rate
            if isinstance(r.rate, Constant):
                return get_by_name(p, r.rate)
            if isinstance(r.rate, Parameter) and isinstance(r.rate.default, Number):
                return p[r.rate]
            # Check if the rate can be compiled to a constant
            if not depends_on_at_least_one_variable_or_time(r.rate):
                rate = substitute(
                    r.rate,
                    {
                        named: replace_algebraic_expressions(named, p)
                        for named in yield_named(r.rate)
                        if isinstance(named, Parameter | Constant)
                    },
                )
                return float(translate(rate, libstd))

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
