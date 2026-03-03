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
        other_values: Mapping[Components, Initial] | None = None,
        save_at: Iterable[ArrayLike],
    ):
        other_values = {} if other_values is None else other_values
        solutions = []
        for v in values:
            result = self.func(
                sim.solve(save_at=save_at, values={parameter: v} | other_values)
            )
            solutions.append(result)
        data_arrays = {}
        if isinstance(solutions[0], Mapping):
            for k in solutions[0].keys():
                data_arrays[k] = xr.DataArray(
                    np.array([s[k] for s in solutions]),
                    dims=str(parameter),
                    coords={str(parameter): values},
                )
        else:
            data_arrays["result"] = xr.DataArray(
                np.array(solutions),
                dims=str(parameter),
                coords={str(parameter): values},
            )
        return xr.Dataset(data_arrays)
