from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from .. import solvers
from ..analysis.period_methods import autoperiod, fft_peak
from ..simulator import Components, Simulator
from ..types import Initial, Variable


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
        result = sim.solve(
            values=values,
            solver=self.solver,
            save_at=save_at,
        )
        output = {
            var: self.process_result(
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
        }
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
    ) -> xr.Dataset:
        timestep = T_min / Dt_in_T
        t_end = T_r + T_after_rel * T_max
        save_at = np.arange(0, t_end + (T_after_rel + 0.5) * timestep, timestep)
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
                T_r=T_r,
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
        # fft of result normalized by its mean
        series = np.asarray(series)
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
    return np.sqrt(np.mean((np.asarray(series1) - np.asarray(series2)) ** 2))
