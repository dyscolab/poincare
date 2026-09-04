from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from warnings import warn

import numpy as np
import pint
import xarray as xr

from .. import solvers
from ..analysis.period_methods import autoperiod, fft_peak
from ..simulator import Components, Simulator
from ..types import Initial

ureg = pint.get_application_registry()


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
    ) -> Mapping[Components, tuple[float, float, float]]:
        result = sim.with_values(values).solve(
            save_at=save_at,
        )
        # TODO: How should units be handled? (currently stripping them with dequantify())
        output = {
            var: self.process_result(
                series=np.array(result[str(var)].pint.dequantify().values),
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
        }
        return output

    def sweep(
        self,
        sim: Simulator,
        /,
        *,
        T_min: float | pint.Quantity,
        T_max: float | pint.Quantity,
        rel_time: float | pint.Quantity,
        variables: Components | Iterable[Components] | None = None,
        values: Iterable[Initial],
        parameter: Components,
        method: str = "autoperiod",
        T_after_rel: int = 10,  # periods simulated after relaxation
        timesteps_in_T: int = 10,  # number of timesteps in min period
    ) -> xr.Dataset:
        timestep = T_min / timesteps_in_T
        t_end = rel_time + T_after_rel * T_max
        save_at = pint_arange(
            0,
            t_end + (T_after_rel + 0.5) * timestep,
            timestep,
            target_dimensionality=sim.model._independent_units,
        )  # Units compatible wrapper for np.arange
        if variables is None:
            try:
                used_vars = list(sim.compiled.variables)
            except AttributeError:
                used_vars = list(type(sim.model).variables.index)
        elif isinstance(variables, Iterable):
            used_vars = list(variables)
        else:
            used_vars = [variables]
        results = [
            self.find_period(
                sim,
                values={parameter: v},
                save_at=save_at,
                used_vars=used_vars,
                parameter=parameter,
                T_r=rel_time,
                timestep=timestep,
                method=method,
                T_after_rel=T_after_rel,
                T_min=T_min,
                T_max=T_max,
            )
            for v in values
        ]
        return xr.Dataset(
            {
                str(var): xr.DataArray(
                    data=np.array([results[i][var] for i, v in enumerate(values)]),
                    dims=[str(parameter), "quantity"],
                    coords={
                        str(parameter): values,
                        "quantity": ["period", "amplitude", "difference_rms"],
                    },
                )
                for var in used_vars
            }
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
        series = np.asarray(series)
        data = (series - np.mean(series))[int(np.ceil(T_r / timestep)) :]
        T, verified = period_finder(data, timestep)
        if not verified:
            warn(
                f"could not verifiy period for {variable} with {parameter} = {values[parameter]}, returning period with maximum power"
            )
        if T_max >= T >= T_min:
            T_Dt = (
                round(T / timestep).magnitude
                if isinstance(T, pint.Quantity)
                else round(T / timestep)
            )
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
            T, A, difference_rms = (
                getattr(x, "magnitude", x) for x in (T, A, difference_rms)
            )  # TODO: How should units be handled (the current multidimensional output won't work with different units)
            return (T, A, difference_rms)
        elif T > 0:
            warn(
                f"Period out of range for {parameter} = {values[parameter]}, returning -1 for amplitude and diffence rms"
            )
            T = getattr(T, "magnitude", T)
            return (T, -1, -1)
        else:
            warn(
                f"could not find period for {parameter} = {values[parameter]}, returning -1"
            )
            return (-1, -1, -1)


def mean_quad_dif(series1: Iterable[float], series2: Iterable[float]) -> float:
    return np.sqrt(np.mean((np.asarray(series1) - np.asarray(series2)) ** 2))


def pint_arange(start, stop, step, target_dimensionality=None):
    target_unit = None
    for arg in (start, stop, step):
        if isinstance(arg, pint.Quantity):
            target_unit = arg.units
            break
    if getattr(target_unit, "dimensionality", None) != target_dimensionality:
        raise pint.PintError(
            f"Target dimensionality {target_unit.dimensionality} is not compatible with the dimensionality of the system's independent variable {target_dimensionality}."
        )

    def to_magnitude(val):
        if hasattr(val, "units"):
            if target_unit is None:
                return val.magnitude
            return val.to(target_unit).magnitude
        return val

    start_m = to_magnitude(start)
    stop_m = to_magnitude(stop)
    step_m = to_magnitude(step)

    magnitude_array = np.arange(start_m, stop_m, step_m)

    if target_unit is not None:
        return magnitude_array * target_unit

    return magnitude_array
