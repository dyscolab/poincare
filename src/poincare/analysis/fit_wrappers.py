"""Shortcuts for common fits with a single target, built on Fitter."""

from collections.abc import Mapping

import xarray as xr

from ..simulator import Components, Simulator
from ..types import Initial, Number
from .fits import Fitter, FitResult, UnitsHandler
from .steady_state import SteadyState

Output = xr.DataArray | xr.Dataset


def fit_result(
    sim: Simulator,
    results: Output,
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    sigma: Output | None = None,
    **kwargs,
) -> FitResult:
    """Fit a time course: results is compared to sim.solve at the times of
    its first coordinate.

    sigma is the experimental error, of the same form as results. Extra
    keyword arguments are passed to scipy.optimize.least_squares.
    """
    save_at = UnitsHandler.get_save_at(results)
    fitter = Fitter(
        sim,
        [lambda sim: sim.solve(save_at=save_at)],
        [results],
        p0=p0,
        scales=1 if scale is None else scale,
        sigma=None if sigma is None else [sigma],
    )
    return fitter.fit(**kwargs)


def fit_steady_state(
    sim: Simulator,
    steady: SteadyState,
    variable: Components,
    results: Output,
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    sigma: Output | None = None,
    **kwargs,
) -> FitResult:
    """Fit steady states: results is compared to steady.sweep over variable,
    at the values of its first coordinate.

    sigma is the experimental error, of the same form as results. Extra
    keyword arguments are passed to scipy.optimize.least_squares.
    """
    values = UnitsHandler.get_save_at(results)
    fitter = Fitter(
        sim,
        [lambda sim: steady.sweep(sim, variable=variable, values=values)],
        [results],
        p0=p0,
        scales=1 if scale is None else scale,
        sigma=None if sigma is None else [sigma],
    )
    return fitter.fit(**kwargs)
