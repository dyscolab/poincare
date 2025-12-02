from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from numpy.typing import ArrayLike, NDArray
from scipy_events import Event, Events, SmallDerivatives
from scipy_events.typing import Condition

from ..simulator import Simulator, Components
from .. import solvers
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
        return (
            sim.solve(
                values=values,
                solver=self.solver,
                t_span=(0, self.t_end),
                save_at=(self.t_end,),
                events=[Event(condition=self.condition, terminal=True)],
            )
            .reset_index()
            .iloc[0]
        )

    def sweep(
        self,
        sim: Simulator,
        /,
        *,
        variable: Components,
        values: Iterable[Initial],
    ):
        return pd.DataFrame(
            data=[self.solve(sim, values={variable: v}) for v in values],
            index=pd.Series(values, name=variable),
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

            def update_from_result(self, result: pd.Series):
                for k in self.keys():
                    try:
                        self[k] = result[str(k)]
                    except KeyError:
                        pass

        current_values = Values()
        result = pd.Series()
        output = {}
        for direction, values in zip(names, (values, reversed(values))):
            for v in values:
                current_values.update_from_problem({variable: v})
                current_values.update_from_result(result)
                output[(direction, v)] = result = self.solve(sim, values=current_values)
        df = pd.DataFrame.from_dict(output, orient="index")
        df.index.names = ("direction", str(variable))
        return df

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
        df = self.sweep_up_and_down(
            sim,
            variable=variable,
            values=values,
            names=(up, down),
        )
        df = df.drop(columns=["time", "event"])
        sum = (df.loc[up] + df.loc[down]).abs()
        diff = (df.loc[up] - df.loc[down]).abs()
        return (diff > atol) | (diff > rtol * sum)
