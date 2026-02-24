from __future__ import annotations

import numbers
from collections import ChainMap
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pint
import pint_xarray
import xarray as xr
from numpy.typing import ArrayLike
from scipy_events import Events
from symbolite.core.value import Value

from . import solvers
from ._node import Node
from ._utils import eval_content
from .compile import (
    RHS,
    Backend,
    SystemCompiler,
    Transform,
    compile_transform,
    depends_on_at_least_one_variable_or_time,
)
from .types import (
    Array1d,
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
    y: Array1d
    p: Array1d
    transform: Transform
    scale: Sequence[Number | pint.Quantity[Any]]


def rescale(q: Number | pint.Quantity[Any]) -> Number:
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
        compiler = SystemCompiler(self.model, backend=backend)
        self.compiled = compiler.compiled
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
            ds = xr.Dataset(
                {
                    k: xr.DataArray(
                        data=x * s.magnitude, dims="time", coords={"time": t}
                    ).pint.quantify(
                        s.units, pint_xarray.setup_registry(s.units._REGISTRY)
                    )
                    if isinstance(s, pint.Quantity)
                    else xr.DataArray(data=x * s, dims="time", coords={"time": t})
                    for k, s, x in zip(self.transform.output.keys(), problem.scale, y.T)
                }
            )
            for s in problem.scale:
                if isinstance(s, pint.Quantity):
                    s.units._REGISTRY.force_ndarray_like = False
            return ds

        ds = _convert(solution.t, solution.y)

        if len(events) > 0:
            ds_events = (
                _convert(t, y).assign(event=i)
                for i, (t, y) in enumerate(zip(solution.t_events, solution.y_events))
            )
            ds = xr.concat(
                [ds.assign(event=np.nan), *ds_events], "time", data_vars="all"
            )
        return ds

    def interact(
        self,
        values: Mapping[Components, tuple[float, ...] | ipywidgets.Widget]
        | Sequence[Components] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        save_at: ArrayLike,
        func: Callable[[xr.Dataset], Any] = lambda ds: ds.to_dataframe().plot(),
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
