from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from ..simulator import Components, Simulator
from ..types import Any, Initial


@dataclass(frozen=True)
class Sweeper:
    """
    Class to do a parameter sweep and apply a function on the output for each result.
    """

    func: Callable[xr.Dataset, Mapping[str, Any] | Any]

    def sweep(
        self,
        sim: Simulator,
        *,
        parameter: Components,
        values: Iterable[Initial],
        save_at: Iterable[ArrayLike],
        unpack: bool = True,
    ):
        solutions = []
        for v in values:
            result = self.func(sim.with_values({parameter: v}).solve(save_at=save_at))
            solutions.append(result)
        data_arrays = {}
        if isinstance(solutions[0], dict) and unpack:
            try:
                for k in solutions[0].keys():
                    data_arrays[k] = xr.DataArray(
                        np.asarray([s[k] for s in solutions]),
                        dims=str(parameter),
                        coords={str(parameter): values},
                    )
            except ValueError:
                for k in solutions[0].keys():
                    data_arrays[k] = xr.DataArray(
                        np.asarray([s[k] for s in solutions], dtype=object),
                        dims=str(parameter),
                        coords={str(parameter): values},
                    )
        else:
            try:
                data_arrays["result"] = xr.DataArray(
                    np.asarray(solutions),
                    dims=str(parameter),
                    coords={str(parameter): values},
                )
            except ValueError:
                data_arrays["result"] = xr.DataArray(
                    np.asarray(solutions, dtype=object),
                    dims=str(parameter),
                    coords={str(parameter): values},
                )
        return xr.Dataset(data_arrays)
