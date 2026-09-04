from collections.abc import Mapping, Sequence

import numpy as np
import pint
import xarray as xr
from scipy.optimize import least_squares

from .._node import Node
from ..simulator import Components, Simulator
from ..types import Initial, Number


def get_default_or_inital(obj: Node) -> Initial:
    try:
        return obj.default
    except AttributeError:
        return obj.initial


class UnitsHandler:
    def __init__(self, fit_variables, fit_parameters, results, clean_p0) -> None:
        self.var_units = {}
        self.param_units = {}

        for var in fit_variables:
            res = results[var]
            if isinstance(res, pint.Quantity):
                self.var_units[var] = res.pint.units
            else:
                self.var_units[var] = None

        for param in fit_parameters:
            default = clean_p0.get(param, get_default_or_inital(param))
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
            try:
                return obj.to(self.var_units[var]).magnitude
            except pint.DimensionalityError as err:
                raise pint.PintError(f"Unexpected units in {var}: {err}")
            except AttributeError:
                raise pint.PintError(f"units given in {var} when none were excpected")
        else:
            if self.var_units[var] is not None:
                raise (
                    pint.PintError(
                        f"No units given for variable {var}, expected dimensionality {self.var_units[var].dimensionality}"
                    )
                )
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
            try:
                return obj.to(self.param_units[param]).magnitude
            except pint.DimensionalityError as err:
                raise pint.PintError(f"Unexpected units in {param}: {err}")
            except AttributeError:
                raise pint.PintError(f"units given in {param} when none were excpected")
        else:
            return obj

    def dequantify_result(self, obj: xr.DataArray, var: Components) -> xr.DataArray:
        if obj.pint.units is not None:
            return obj.pint.to(self.var_units[var]).pint.dequantify().to_numpy()
        else:
            return obj.to_numpy()

    def get_save_at(self, ds: xr.Dataset) -> Sequence | pint.Quantity:
        index = ds[next(iter(ds.coords))]
        if index.pint.units is not None:
            return index.pint.dequantify().to_numpy() * index.pint.units
        else:
            return index.to_numpy()

    def dequantify_bounds(
        self,
        bounds: Mapping[Components, tuple[Initial, Initial]],
        param: Components,
        i: int,
    ) -> tuple[Number]:
        try:
            return self.dequantify_parameter(bounds[param][i], param)
        except KeyError:
            return (-1) ** (i + 1) * np.inf


def fit_result(
    sim: Simulator,
    results: xr.Dataset,
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    **kwargs,
):
    fit_variables = list(results.keys())
    if p0:
        fit_parameters = list(p0.keys())
    else:
        fit_parameters = list(sim.model.parameters.values)
    clean_p0, bounds = parse_p0(p0)
    if scale is not None:
        scale = np.array([[scale.get(var, 1)] for var in fit_variables])
    else:
        scale = np.array([[1] for var in fit_variables])
    units = UnitsHandler(fit_variables, fit_parameters, results, clean_p0)
    bounds = tuple(
        np.array(
            [
                units.dequantify_bounds(bounds=bounds, param=param, i=i)
                for param in fit_parameters
            ]
        )
        for i in range(2)
    )
    y0 = np.array([units.dequantify_result(results[var], var) for var in fit_variables])
    save_at = units.get_save_at(results)

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
                clean_p0.get(param, get_default_or_inital(param)), param
            )
            for param in fit_parameters
        ]
    )
    solution = least_squares(f, x0, bounds=bounds, **kwargs)
    formatted_solution = {
        var: units.quantify_parameter(solution.x[i], var)
        for i, var in enumerate(fit_parameters)
    }
    return formatted_solution


def parse_p0(
    p0: Mapping[Components, None | Initial | Sequence[Initial]],
) -> tuple[Mapping[Components, Initial], Mapping[Components, tuple[Initial, Initial]]]:
    clean_p0 = {}
    bounds = {}
    is_tuple = False
    for param, value in p0.items():
        try:
            n = len(value)
            is_tuple = True
        except TypeError:
            is_tuple = False

        if is_tuple:
            if n != 3:
                raise TypeError(
                    "p0 values must be None, an initial condition or a tuple (initial condition, lower bound, upper bound)"
                )
            if value[0] is not None:
                clean_p0[param] = value[0]
            bounds[param] = (value[1], value[2])
        elif value is not None:
            clean_p0[param] = value
    return clean_p0, bounds


def make_target(
    results: Mapping[Components | str, Sequence | pint.Quantity],
    save_at: Sequence | pint.Quantity,
) -> xr.Dataset:
    ureg = pint.get_application_registry()

    ureg.force_ndarray_like = True

    try:
        data_vars = {}

        if isinstance(save_at, pint.Quantity):
            units_map = {"time": save_at.units}
            dequantified_save_at = save_at.magnitude

        else:
            units_map = {}
            dequantified_save_at = save_at

        for key, value in results.items():
            if isinstance(value, pint.Quantity):
                units_map[key] = value.units
                dequantified_value = value.magnitude
            else:
                dequantified_value = value

            data_vars[key] = xr.DataArray(
                dequantified_value,
                dims=["time"],
            )

        ds = xr.Dataset(data_vars=data_vars, coords={"time": dequantified_save_at})

        ds = ds.pint.quantify(units_map)

    finally:
        ureg.force_ndarray_like = False

    return ds
