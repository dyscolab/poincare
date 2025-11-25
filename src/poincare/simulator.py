from __future__ import annotations

import numbers
from collections import ChainMap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Mapping, Sequence
from warnings import warn

import numpy as np
import pandas as pd
import pint
import pint_pandas
from numpy.typing import ArrayLike, NDArray
from scipy_events import Event, Events, SmallDerivatives
from scipy_events.typing import Condition
from symbolite.core.value import Value

from . import solvers
from ._node import Node
from ._utils import eval_content
from .compile import (
    RHS,
    Backend,
    Transform,
    compile_diffeq,
    compile_transform,
    depends_on_at_least_one_variable_or_time,
)
from .period_methods import autoperiod, fft_peak
from .types import (
    Constant,
    Derivative,
    Initial,
    Number,
    Parameter,
    System,
    Variable,
)

if TYPE_CHECKING:
    import ipywidgets

Components = Constant | Parameter | Variable | Derivative


@dataclass
class Problem:
    rhs: RHS
    t: tuple[float, float]
    y: Sequence[Number]
    p: Sequence[Number]
    transform: Transform
    scale: Sequence[Number | pint.Quantity]


def rescale(q: Number | pint.Quantity) -> Number:
    if isinstance(q, pint.Quantity):
        return q.to_base_units().magnitude
    else:
        return q


def get_scale(q: Number | pint.Quantity) -> Number | pint.Quantity:
    if isinstance(q, pint.Quantity):
        unit = q.units
        scale = (1 / unit).to_base_units().magnitude
        return scale * unit
    else:
        return 1


class Simulator:
    def __init__(
        self,
        system: System | type[System],
        /,
        *,
        backend: Backend = "numpy",
        transform: Sequence[Value] | Mapping[Hashable, Value] | None = None,
    ):
        self.model = system
        self.compiled = compile_diffeq(system, backend)
        self.transform = self._compile_transform(transform)

    def _compile_transform(
        self,
        transform: Sequence[Value] | Mapping[Hashable, Value] | None,
    ):
        if isinstance(transform, Sequence):
            transform = {str(x): x for x in transform}
        return compile_transform(self.model, self.compiled, transform)

    def create_problem(
        self,
        values: Mapping[Components, Initial | Value] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        transform: Sequence[Value] | Mapping[Hashable, Value] | None = None,
    ):
        if transform is None:
            compiled_transform = self.transform
        else:
            compiled_transform = self._compile_transform(transform)

        if any(
            depends_on_at_least_one_variable_or_time(self.compiled.mapper[k])
            or depends_on_at_least_one_variable_or_time(v)
            for k, v in values.items()
        ):
            raise ValueError("must recompile to change time-dependent assignments")

        for k, v in values.items():
            default = self.compiled.mapper[k]
            match [v, default]:
                case [pint.Quantity() as q1, pint.Quantity() as q2]:
                    if not q1.is_compatible_with(q2):
                        raise pint.DimensionalityError(
                            q1.units, q2.units, extra_msg=f" for {k}"
                        )
                case [pint.Quantity() as q, _] | [_, pint.Quantity() as q]:
                    if not q.dimensionless:
                        raise pint.DimensionalityError(
                            q.units, pint.Unit("dimensionless"), extra_msg=f" for {k}"
                        )

        content = ChainMap(
            values,
            self.compiled.mapper,
            self.transform.output,
            {self.compiled.independent[0]: t_span[0]},
        )
        assert self.compiled.libsl is not None
        result = eval_content(
            content,
            self.compiled.libsl,
            is_root=lambda x: isinstance(x, Number | pint.Quantity),
            is_dependency=lambda x: isinstance(x, Node),
        )
        y0 = np.fromiter(
            map(rescale, (result[k] for k in self.compiled.variables)),
            dtype=float,
            count=len(self.compiled.variables),
        )
        p0 = np.fromiter(
            map(rescale, (result[k] for k in self.compiled.parameters)),
            dtype=float,
            count=len(self.compiled.parameters),
        )
        scale = [get_scale(result[k]) for k in self.transform.output.keys()]
        return Problem(
            rhs=self.compiled.func,
            t=t_span,
            y=y0,
            p=p0,
            transform=compiled_transform.func,
            scale=scale,
        )

    def solve(
        self,
        values: Mapping[Components, Initial | Value] = {},
        *,
        t_span: tuple[float, float] | None = None,
        save_at: ArrayLike | None = None,
        solver: solvers.Solver = solvers.LSODA(),
        events: Sequence[Events] = (),
    ):
        if save_at is not None:
            save_at = np.asarray(save_at)

        if t_span is None:
            if save_at is None:
                raise TypeError("must provide t_span and/or save_at.")
            t_span = (0, save_at[-1])

        problem = self.create_problem(values, t_span=t_span)
        solution = solver(problem, save_at=save_at, events=events)

        def _convert(t, y):
            return pd.DataFrame(
                {
                    k: pint_pandas.PintArray(
                        x * s.magnitude, pint_pandas.PintType(s.units)
                    )
                    if isinstance(s, pint.Quantity)
                    else x * s
                    for k, s, x in zip(self.transform.output.keys(), problem.scale, y.T)
                },
                index=pd.Series(t, name="time"),
            )

        df = _convert(solution.t, solution.y)
        if len(events) > 0:
            df_events = (
                _convert(t, y).assign(event=i)
                for i, (t, y) in enumerate(zip(solution.t_events, solution.y_events))
            )
            df = pd.concat([df.assign(event=pd.NA), *df_events])
        return df

    def interact(
        self,
        values: Mapping[Components, tuple[float, ...] | ipywidgets.Widget]
        | Sequence[Components] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        save_at: ArrayLike,
        func: Callable[[pd.DataFrame], Any] = lambda df: df.plot(),
    ):
        try:
            import ipywidgets
        except ImportError:
            raise ImportError(
                "must install ipywidgets to use interactuve."
                " Run `pip install ipywidgets`."
            )

        if len(values) == 0:
            values = self.compiled.mapper
        elif isinstance(values, Sequence):
            values = {k: self.compiled.mapper[k] for k in values}

        name_map = {}
        value_map = {}
        unit_map = {}
        for k, v in values.items():
            unit = 1
            match v:
                case numbers.Real() as default:
                    if v == 0:
                        v = 1
                    min, max = sorted((default / 10, v * 10))
                    widget = ipywidgets.FloatSlider(
                        default,
                        min=min,
                        max=max,
                        step=0.1 * abs(v),
                    )
                case pint.Quantity(magnitude=v, units=unit):
                    min, max = sorted((v / 10, v * 10))
                    widget = ipywidgets.FloatSlider(
                        v, min=min, max=max, step=0.1 * abs(v)
                    )
                case (min, max, step):
                    v = self.compiled.mapper[k]
                    if not isinstance(v, Number):
                        v = None
                    widget = ipywidgets.FloatSlider(v, min=min, max=max, step=step)
                case ipywidgets.Widget():
                    widget = v
                case _:
                    continue

            name = str(k)
            name_map[name] = k
            value_map[name] = widget
            unit_map[name] = unit

        def solve_and_plot(**kwargs):
            result = self.solve(
                {name_map[k]: v * unit_map.get(k, 1) for k, v in kwargs.items()},
                t_span=t_span,
                save_at=save_at,
            )
            func(result)

        ipywidgets.interact(solve_and_plot, **value_map)


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
            atol = factor * self.solver.atol
        if rtol is None:
            rtol = factor * self.solver.rtol

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


@dataclass(kw_only=True, frozen=True)
class Oscillations:
    solver: solvers.Solver = solvers.LSODA()

    def find_period(
        self,
        sim: Simulator,
        /,
        *,
        save_at: Sequence[float],
        values: Mapping[Components, Initial],
        used_vars: Iterable[Components],
        parameter: Components,
        T_r: float,
        timestep: float,
        method: str = "autoperiod",
        T_after_rel: int,
        T_min: float,
        T_max: float,
    ) -> tuple[float, float]:
        result = sim.solve(
            values=values,
            solver=self.solver,
            save_at=save_at,
        )
        output_tuples = [
            self.process_result(
                series=np.array(result[str(var)].values),
                variable=var,
                values=values,
                parameter=parameter,
                T_r=T_r,
                timestep=timestep,
                method=method,
                T_after_rel=T_after_rel,
                T_min=T_min,
                T_max=T_max,
            )
            for var in used_vars
        ]
        output = [x for t in output_tuples for x in t]
        return output

    def sweep(
        self,
        sim: Simulator,
        /,
        *,
        T_min: float,
        T_max: float,
        T_r: float,
        variables: Components | Iterable[Components] | None = None,
        values: Iterable[Initial],
        parameter: Components,
        method: str = "autoperiod",
        T_after_rel: int = 10,  # periods simulated after relaxation
        Dt_in_T: int = 10,  # number of timesteps in min period
    ) -> pd.DataFrame:
        timestep = T_min / Dt_in_T
        t_end = T_r + T_after_rel * T_max
        save_at = np.arange(0, t_end + (T_after_rel + 0.5) * timestep, timestep)
        if variables is None:
            used_vars = list(sim.model.variables.index)
        elif isinstance(variables, Iterable):
            used_vars = variables
        else:
            used_vars = [variables]
        if len(used_vars) > 1:
            cols = pd.MultiIndex.from_product(
                [
                    [str(var) for var in used_vars],
                    ["period", "amplitude", "difference_rms"],
                ],
                names=["variable", "quantity"],
            )
        else:
            cols = ["period", "amplitude", "difference_rms"]
        return pd.DataFrame(
            data=[
                self.find_period(
                    sim,
                    values={parameter: v},
                    save_at=save_at,
                    used_vars=used_vars,
                    parameter=parameter,
                    T_r=T_r,
                    timestep=timestep,
                    method=method,
                    T_after_rel=T_after_rel,
                    T_min=T_min,
                    T_max=T_max,
                )
                for v in values
            ],
            index=pd.Series(values, name=parameter),
            columns=cols,
        )

    def process_result(
        self,
        series: Iterable[float],
        parameter: Components,
        variable: Components,
        T_r: float,
        timestep: float,
        T_after_rel: int,
        T_min: float,
        T_max: float,
        values: Mapping[Components, Initial],
        method: str = "autoperiod",
    ):
        methods = {"autoperiod": autoperiod, "fft_peak": fft_peak}
        try:
            period_finder = methods[method]
        except KeyError:
            raise KeyError(f"{method} is  not a valid method")
        # fft of result normalized by its mean
        data = (series - np.mean(series))[int(np.ceil(T_r / timestep)) :]
        T, verified = period_finder(data, timestep)
        if not verified:
            warn(
                f"could not verifiy period for {variable} with {parameter} = {values[parameter]}, returning period with maximum power"
            )
        if T_max >= T >= T_min:
            T_Dt = round(T / timestep)
            periods = np.reshape(data[-T_Dt * T_after_rel :], (T_after_rel, T_Dt))
            A = (
                np.mean(np.max(periods, axis=1) - np.min(periods, axis=1)) / 2
            )  # Mean amplitudes of periods after relaxation
            difference_rms = np.mean(
                [
                    mean_quad_dif(periods[i], periods[i + 1])
                    for i in range(T_after_rel - 1)
                ]
            )
            return (T, A, difference_rms)
        elif T > 0:
            warn(
                f"Period out of range for {parameter} = {values[parameter]}, returning -1 for amplitude and diffence rms"
            )
            return (T, -1, -1)
        else:
            warn(
                f"could not find period for {parameter} = {values[parameter]}, returning -1"
            )
            return (-1, -1, -1)


def mean_quad_dif(series1: Iterable[float], series2: Iterable[float]) -> float:
    return np.sqrt(np.mean((series1 - series2) ** 2))
