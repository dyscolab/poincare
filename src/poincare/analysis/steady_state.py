from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy_events import Event, SmallDerivatives
from scipy_events.typing import Condition

from .. import solvers
from ..simulator import Components, Simulator
from ..types import (
    Initial,
)


@dataclass(kw_only=True, frozen=True)
class SteadyState:
    """Find steady states by running the solver until condition or t_end.

    By default, the condition is that the derivatives are small."""

    solver: solvers.Solver = solvers.LSODA()
    condition: Condition = SmallDerivatives()
    t_end: float = np.finfo(np.float64).max

    def solve(
        self,
        sim: Simulator,
        /,
        *,
        values: Mapping[Components, Initial] = {},
    ):
        return sim.solve(
            values=values,
            solver=self.solver,
            t_span=(0, self.t_end),
            save_at=(self.t_end,),
            events=[Event(condition=self.condition, terminal=True)],
        )

    def sweep(
        self,
        sim: Simulator,
        /,
        *,
        variable: Components,
        values: Iterable[Initial],
    ):
        results = {v: self.solve(sim, values={variable: v}) for v in values}
        return xr.Dataset(
            {
                str(var): xr.DataArray(
                    np.array([results[v][var].item() for v in values]),
                    dims=str(variable),
                    coords={str(variable): values},
                )
                for var in [str(var) for var in sim.compiled.variables]
            }
            | {
                "time": xr.DataArray(
                    np.array([results[v]["time"].item() for v in values]),
                    dims=str(variable),
                    coords={str(variable): values},
                ),
                "event": xr.DataArray(
                    np.array([results[v]["event"].item() for v in values]),
                    dims=str(variable),
                    coords={str(variable): values},
                ),
            }
        )

    def sweep_up_and_down(
        self,
        sim: Simulator,
        /,
        *,
        variable: Components,
        values: Sequence[Initial] | NDArray,
        names: tuple[str, str] = ("up", "down"),
    ):
        class Values(dict):
            def update_from_problem(self, values: dict):
                prob = sim.create_problem(values)
                self.update(zip(sim.compiled.variables, prob.y))
                self.update(zip(sim.compiled.parameters, prob.p))

            def update_from_result(self, result: xr.DataArray):
                for k in self.keys():
                    try:
                        self[k] = result[str(k)].item()
                    except KeyError:
                        pass

        current_values = Values()
        result = xr.DataArray()
        output = {direction: {} for direction in names}
        for direction, vals in zip(names, (values, reversed(values))):
            for v in vals:
                current_values.update_from_problem({variable: v})
                current_values.update_from_result(result)
                output[direction][v] = result = self.solve(sim, values=current_values)
        return xr.Dataset(
            {
                str(var): xr.DataArray(
                    data=np.array(
                        [
                            [
                                output[direction][v][str(var)].item()
                                for direction in names
                            ]
                            for v in values
                        ]
                    ),
                    dims=(str(variable), "direction"),
                    coords={
                        str(variable): values,
                        "direction": np.array(names),
                    },
                )
                for var in sim.compiled.variables
            }
        )

    def bistability(
        self,
        sim: Simulator,
        /,
        *,
        variable: Components,
        values: Sequence[Initial] | NDArray,
        atol: float | None = None,
        rtol: float | None = None,
        factor: float = 1,
    ):
        if atol is None:
            atol: float = np.max(factor * self.solver.atol)
        if rtol is None:
            rtol: float = np.max(factor * self.solver.rtol)

        up, down = "up", "down"
        uad = self.sweep_up_and_down(
            sim,
            variable=variable,
            values=values,
            names=(up, down),
        )
        up = uad.sel(direction="up")
        down = uad.sel(direction="down")
        sum = xr.ufuncs.abs(up + down)
        diff = xr.ufuncs.abs(up - down)
        return (diff > atol) | (diff > rtol * sum)
