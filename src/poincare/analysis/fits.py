from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pint
import xarray as xr
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from .._node import Node
from ..simulator import Components, Simulator
from ..types import Initial, Number, Parameter


def get_default_or_inital(obj: Node) -> Initial:
    try:
        return obj.default
    except AttributeError:
        return obj.initial


class UnitsHandler:
    def __init__(self, fit_variables, fit_parameters, results, p0) -> None:
        self.var_units = {}
        self.param_units = {}

        for var in fit_variables:
            res = results[var]
            if isinstance(res, pint.Quantity):
                self.var_units[var] = res.units
            else:
                self.var_units[var] = None

        for param in fit_parameters:
            default = p0.get(param, get_default_or_inital(param))
            if isinstance(default, pint.Quantity):
                self.param_units[param] = default.units
            else:
                self.param_units[param] = None

    def quantify_variable(
        self, obj: Initial | Sequence[Initial], var: Components | str
    ) -> Initial | Sequence[Initial]:
        unit = self.var_units[var]
        if unit is not None:
            return obj * unit
        else:
            return obj

    def dequantify_variable(
        self, obj: Initial | Sequence[Initial], var: Components | str
    ) -> Initial | Sequence[Initial]:
        if isinstance(obj, pint.Quantity):
            return obj.to(self.var_units[var]).magnitude
        else:
            return obj

    def quantify_parameter(
        self, obj: Initial | Sequence[Initial], param: Components
    ) -> Initial | Sequence[Initial]:
        unit = self.param_units[param]
        if unit is not None:
            return obj * unit
        else:
            return obj

    def dequantify_parameter(
        self, obj: Initial | Sequence[Initial], param: Components
    ) -> Initial | Sequence[Initial]:
        if isinstance(obj, pint.Quantity):
            return obj.to(self.param_units[param]).magnitude
        else:
            return obj

    def dequantify_result(self, obj: xr.DataArray, var: Components) -> xr.DataArray:
        if obj.pint.units is not None:
            return obj.pint.to(self.var_units[var]).pint.dequantify().to_numpy()
        else:
            return obj


# "A": fix
# "A": initial value
# "A": (initial, min, max)

# cruzado por tengo los default


def fit_result(
    sim: Simulator,
    results: Mapping[Components | str, Sequence[Initial]],
    save_at: ArrayLike,
    fit_parameters: Iterable[Components] | None = None,
    p0: Mapping[Components, Initial] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    **kwargs,
):
    fit_variables = list(results.keys())
    if fit_parameters is None:
        fit_parameters = list(
            sim.model._yield(Parameter)
        )  # TODO: infer from p0 instead? Or in addition to this?
    if scale is not None:
        scale = np.array([[scale.get(var, 1)] for var in fit_variables])
    else:
        scale = np.array([[1] for var in fit_variables])

    units = UnitsHandler(fit_variables, fit_parameters, results, p0)

    y0 = np.array(
        [units.dequantify_variable(results[var], var) for var in fit_variables]
    )

    def f(x):
        x = [
            units.quantify_parameter(x[i], param)
            for i, param in enumerate(fit_parameters)
        ]
        result = sim.with_values(
            {fit_parameters[i]: val for i, val in enumerate(x)}
        ).solve(
            save_at=save_at,
        )
        y = np.array(
            [units.dequantify_result(result[str(var)], var) for var in fit_variables]
        )
        res = (y - y0) * scale  # TODO: is * scale or / scale more intuitive?
        return res.ravel()

    x0 = np.array(
        [
            units.dequantify_parameter(
                p0.get(param, get_default_or_inital(param)), param
            )
            for param in fit_parameters
        ]
    )
    solution = least_squares(f, x0, **kwargs)
    formatted_solution = {
        var: units.quantify_parameter(solution.x[i], var)
        for i, var in enumerate(fit_parameters)
    }
    return formatted_solution
