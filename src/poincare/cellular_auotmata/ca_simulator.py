from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping, Hashable
from itertools import product

import numpy as np
from numpy import ndarray
import pandas as pd

from symbolite import Symbol

from .ca_types import Automata, SquareAutomata
from ..types import System
from ..compile import Backend, build_first_order_symbolic_ode, Array, MutableArray
from ..simulator import Simulator, Problem
from ..solvers import LSODA
from ..types import Variable, Derivative, Parameter, Constant, Literal, Number, Initial


@dataclass
class CASimulator:
    # model: Automata
    model: SquareAutomata
    shape: tuple[int]
    backend: Backend = "numpy"
    transform: Sequence[Symbol] | Mapping[Hashable, Symbol] | None = None

    def __post_init__(self):
        self.cell_sim = Simulator(self.model.cell, backend=self.backend)
        self.interact_sim = Simulator(
            self.model.interaction_system, backend=self.backend
        )
        self.boundary_sim = Simulator(self.model.boundary_system, backend=self.backend)

        self.symbolic_cell = build_first_order_symbolic_ode(self.model.cell)

        self.geometry = self.model.geometry(self.shape)
        self.func = self.create_func()

    def create_default_initials(self, variable_index: int, values) -> Array:
        y_0 = self.cell_sim.create_problem().y[variable_index]
        return np.full(np.prod(self.shape), y_0)

    def create_initials(self, values) -> Array:
        vars = self.symbolic_cell.variables
        intials = np.array(
            [
                np.ravel(
                    values.get(
                        vars[i],
                        self.create_default_initials(variable_index=i, values=values),
                    ).astype(float, copy=False)
                )
                for i in range(len(vars))
            ]
        )
        return np.ravel(np.column_stack(intials))

    def create_default_parameters(self, parameter_index: int, values) -> Array:
        p_0 = self.cell_sim.create_problem().p[parameter_index]
        return np.full(np.prod(self.shape), p_0)

    def create_parameters(self, values) -> Array:
        params = self.symbolic_cell.parameters
        try:
            intials = np.array(
                [
                    np.ravel(
                        values.get(
                            params[i], self.create_default_parameters(i, values=values)
                        ).astype(float, copy=False)
                    )
                    for i in range(len(params))
                ]
            )
            return np.ravel(np.column_stack(intials))
        except ValueError:
            return np.array([])

    def create_func(self):
        def func(t: float, y: Array, p: Array, dy: MutableArray):
            for index in self.geometry.yield_locations():
                var_num = len(self.symbolic_cell.variables)
                param_num = len(self.symbolic_cell.parameters)
                one_d_index = self.geometry.get(index)
                one_d_slice = slice(var_num * one_d_index, var_num * (one_d_index + 1))
                parameter_slice = slice(
                    param_num * one_d_index, param_num * (one_d_index + 1)
                )
                dy[one_d_slice] = self.cell_sim.compiled.func(
                    t, y[one_d_slice], p[parameter_slice], dy[one_d_slice]
                )
                neighbors = self.geometry.iter_neighbors(index)
                for neighbor, interaction in neighbors:
                    if interaction == "cell_interaction":
                        neighbor_one_d_index = self.geometry.get(neighbor)
                        neighbor_one_d_slice = slice(
                            var_num * neighbor_one_d_index,
                            var_num * (neighbor_one_d_index + 1),
                        )
                        neighbor_parameter_slice = slice(
                            param_num * neighbor_one_d_index,
                            param_num * (neighbor_one_d_index + 1),
                        )
                        dy[one_d_slice] += self.interact_sim.compiled.func(
                            t,
                            np.concatenate((y[one_d_slice], y[neighbor_one_d_slice])),
                            np.concatenate(
                                (p[parameter_slice], p[neighbor_parameter_slice])
                            ),
                            np.zeros(2 * var_num, dtype=float),
                        )[:var_num]
                    if interaction == "boundary":
                        dy[one_d_slice] += self.boundary_sim.compiled.func(
                            t,
                            y[one_d_slice],
                            p[parameter_slice],
                            np.zeros(var_num, dtype=float),
                        )
            return dy

        return func

    def create_problem(
        self, values: Mapping[Variable | Parameter | Constant, ndarray], t_span
    ):
        y = self.create_initials(values=values)
        p = self.create_parameters(values=values)
        return Problem(
            self.func,
            t_span,
            y,
            p,
            transform=lambda t, y, p, dy: y,
            scale=np.ones_like(y),
        )

    def solve(
        self,
        *,
        save_at: Sequence[Number],
        values: Mapping[Variable | Parameter | Constant, ndarray] | None = None,
        solver=LSODA(),
        format: Literal["dataframe", "array"] = "dataframe",
    ):
        values = values if values is not None else {}
        t_span = (save_at[0], save_at[-1])
        problem = self.create_problem(values=values, t_span=t_span)
        solution = solver(problem, save_at=np.asarray(save_at))
        output = solution.y.reshape(
            np.concatenate(
                [[len(save_at)], self.shape, [len(self.symbolic_cell.variables)]]
            )
        )
        if format == "dataframe":
            cols = pd.MultiIndex.from_product(
                [range(s) for s in self.shape]
                + [[str(var) for var in self.symbolic_cell.variables]]
            )
            return pd.DataFrame(
                output.reshape(
                    len(save_at),
                    np.prod(self.shape) * len(self.symbolic_cell.variables),
                ),
                columns=cols,
            )
        if format == "array":
            output = solution.y.reshape(
                np.concatenate(
                    [[len(save_at)], self.shape, [len(self.symbolic_cell.variables)]]
                )
            )
            return output

    def solve_and_animate(
        self,
        *,
        save_at: Sequence[Number],
        values: Mapping[Variable | Parameter | Constant, ndarray] | None = None,
        solver=LSODA(),
        variable: Variable,
        timescale: Number = 1,
        save_to: str | None = None,
        **kwargs,
    ):
        if len(self.shape) != 2:
            raise ValueError("only 2d automata can be animated")
        full_result = self.solve(
            save_at=save_at, values=values, solver=solver, format="array"
        )
        var_index = self.symbolic_cell.variables.index(variable)
        var_num = len(self.symbolic_cell.variables)
        result = full_result[..., var_index::var_num].reshape(
            np.concatenate([[len(save_at)], self.shape])
        )
        interval = (save_at[1] - save_at[0]) * timescale * 1000
        animation = animate_array(result, interval, **kwargs)
        if isinstance(save_to, str):
            animation.save(save_to, writer="pillow")
        else:
            return animation


def animate_array(frames, interval=100, **kwargs):
    """
    frames: numpy array of shape (n_frames, H, W)
    interval: delay between frames in ms
    """
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], animated=True, **kwargs)
    ax.axis("off")
    ax.set_frame_on(True)

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

    return ani
