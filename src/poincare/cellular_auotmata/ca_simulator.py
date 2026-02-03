from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from numpy import ndarray
from symbolite import Symbol, real

from ..compile import Array, MutableArray
from ..simulator import Components, Problem
from ..solvers import LSODA
from ..types import (
    Constant,
    Literal,
    Number,
    Parameter,
    Variable,
)
from .ca_types import (
    ArbitraryInitial,
    Automata,
    CellType,
    Location,
)


@dataclass
class CASimulator:
    model: Automata
    shape: tuple[int]
    transform: Sequence[Symbol] | Mapping[Hashable, Symbol] | None = None

    def __post_init__(self):
        self.geometry = self.model.geometry(self.shape)
        self.func = self.create_func()

    def calculate_slice(
        self, location: Location, item: Literal["variable", "parameter"]
    ) -> slice:
        index, cell, cell_category_index = self.geometry.get(location)
        item_num = self.model.compiled_cells[cell][index].symbolic.__getattribute__(
            item
        )
        cell_counts = self.geometry.cell_counts
        indices_up_to_cell_type = 0
        cells_up_to_cell_type = 0
        break_outer = False

        for cell_category, cell_compilers in self.model.compiled_cells.items():
            for i, compiler in enumerate(cell_compilers):
                if cell_category == cell and i == index:
                    break_outer = True
                    break
                cells_up_to_cell_type += cell_counts[cell_category][i]
                indices_up_to_cell_type += cell_counts[cell_category][
                    i
                ] * compiler.symbolic.__getattribute__(item)
            if break_outer:
                break

        cell_start = (cell_category_index - cells_up_to_cell_type) * item_num
        return slice(cell_start, cell_start + item_num)

    def create_func(self):
        def func(t: float, y: Array, p: Array, dy: MutableArray):
            for location, cell_category, index in self.geometry.yield_locations():
                variable_slice = self.calculate_slice(
                    location=location, item="variable"
                )
                var_num = variable_slice.stop - variable_slice.start
                parameter_slice = self.calculate_slice(
                    location=location, item="parameter"
                )

                dy[variable_slice] = self.model.compiled_cells[cell_category][
                    index
                ].compiled.func(
                    t, y[variable_slice], p[parameter_slice], dy[variable_slice]
                )

                for (
                    neighbor_location,
                    interaction,
                    interaction_index,
                ) in self.geometry.yield_neighbors(location):
                    if neighbor_location is None:
                        dy[variable_slice] += self.model.compiled_interactions[
                            interaction
                        ][interaction_index].compiled.func(
                            t,
                            y[variable_slice],
                            p[parameter_slice],
                            np.zeros(var_num, dtype=float),
                        )
                    else:
                        neighbor_variable_slice = self.calculate_slice(
                            location=neighbor_location, item="variable"
                        )
                        neighbor_parameter_slice = self.calculate_slice(
                            location=neighbor_location, item="parameter"
                        )
                        dy[variable_slice] += self.model.compiled_interactions[
                            interaction
                        ][index].compiled.func(
                            t,
                            np.concatenate(
                                (y[variable_slice], y[neighbor_variable_slice])
                            ),
                            np.concatenate(
                                (p[parameter_slice], p[neighbor_parameter_slice])
                            ),
                            np.zeros(
                                var_num
                                + (
                                    neighbor_variable_slice.stop
                                    - neighbor_variable_slice.start
                                ),
                                dtype=float,
                            ),
                        )[:var_num]
            return dy

        return func

    def create_problem(
        self,
        values: Mapping[
            CellType, Sequence[Mapping[Components | real.Real, ArbitraryInitial[Array]]]
        ],
        t_span,
    ):
        y = self.geometry.create_initials(
            compiled_cells=self.model.compiled_cells,
            values=values,
        )
        p = self.geometry.create_parameters(
            compiled_cells=self.model.compiled_cells,
            values=values,
        )
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
        values: Mapping[
            CellType,
            Mapping[Components | real.Real, ArbitraryInitial]
            | Sequence[Mapping[Components | real.Real, ArbitraryInitial]],
        ]
        | Sequence[Mapping[Components | real.Real, ArbitraryInitial]]
        | Mapping[Components | real.Real, ArbitraryInitial]
        | None = None,
        solver=LSODA(),
        format: Literal["dataframe", "array"] = "dataframe",
    ):
        if values is None:
            values = {
                cell_category: [{} for x in range(len(system_list))]
                for cell_category, system_list in self.model.cell_systems.items()
            }
        elif isinstance(values, Mapping[Components | real.Real, Any]):
            if (
                len(self.model.cell_systems.keys()) == 1
                and len(list(self.model.cell_systems.values())[0]) == 1
            ):
                values = {
                    list(self.model.cell_systems.keys())[0]: [
                        values,
                    ]
                }
            else:
                raise TypeError(
                    "Must explicit cell_category of initial for automata with more than one cell category, use format values = {cell_category: [initials_for_cell_type]}"
                )
        elif isinstance(values, Sequence[Mapping[Components | real.Real, Any]]):
            if len(self.model.cell_systems.keys()) == 1:
                if len(list(self.model.cell_systems.values())[0]) == len(values):
                    values = {list(self.model.cell_systems.keys())[0]: values}
                else:
                    raise TypeError(
                        "Cell category has more cell_types than elements in values, try padding values with (possibly empty) dictionaries."
                    )
            else:
                raise TypeError(
                    "Must explicit cell_category of initial for automata with more than one cell category, use format values = {cell_category: [initials_for_cell_type]}"
                )
        elif isinstance(
            values,
            Mapping[
                CellType,
                Mapping[Components | real.Real, Any]
                | Sequence[Mapping[Components | real.Real, Any]],
            ],
        ):
            values = {
                cell_category: values.get(cell_category, [{} for i in len(system_list)])
                for cell_category, system_list in self.model.cell_systems.items()
            }
        # TODO: if values is not detected as valid format fail? Or try passing it anyways?
        t_span = (save_at[0], save_at[-1])
        problem = self.create_problem(values=values, t_span=t_span)
        solution = solver(problem, save_at=np.asarray(save_at))
        return self.geometry.format_outupt(solution, format)
        # TODO: fix output formatting
        # output = solution.y.reshape(
        #     np.concatenate(
        #         [[len(save_at)], self.shape, [len(self.symbolic_cell.variables)]]
        #     )
        # )
        # if format == "dataframe":
        #     cols = pd.MultiIndex.from_product(
        #         [range(s) for s in self.shape]
        #         + [[str(var) for var in self.symbolic_cell.variables]]
        #     )
        #     return pd.DataFrame(
        #         output.reshape(
        #             len(save_at),
        #             np.prod(self.shape) * len(self.symbolic_cell.variables),
        #         ),
        #         columns=cols,
        #     )
        # if format == "array":
        #     output = solution.y.reshape(
        #         np.concatenate(
        #             [[len(save_at)], self.shape, [len(self.symbolic_cell.variables)]]
        #         )
        #     )

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
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], animated=True, **kwargs)
    ax.axis("off")
    ax.set_frame_on(True)

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

    return ani


# @dataclass
# class SquareCASimulator:
#     # model: Automata
#     model: SquareAutomata
#     shape: tuple[int]
#     backend: Backend = "numpy"
#     transform: Sequence[Symbol] | Mapping[Hashable, Symbol] | None = None

#     def __post_init__(self):
#         self.cell_sim = Simulator(self.model.cell, backend=self.backend)
#         self.interact_sim = Simulator(
#             self.model.interaction_system, backend=self.backend
#         )
#         self.boundary_sim = Simulator(self.model.boundary_system, backend=self.backend)

#         self.symbolic_cell = build_first_order_symbolic_ode(self.model.cell)

#         self.geometry = self.model.geometry(self.shape)
#         self.func = self.create_func()

#     def create_default_initials(self, variable_index: int, values) -> Array:
#         y_0 = self.cell_sim.create_problem().y[variable_index]
#         return np.full(np.prod(self.shape), y_0)

#     def create_initials(self, values) -> Array:
#         vars = self.symbolic_cell.variables
#         intials = np.array(
#             [
#                 np.ravel(
#                     values.get(
#                         vars[i],
#                         self.create_default_initials(variable_index=i, values=values),
#                     ).astype(float, copy=False)
#                 )
#                 for i in range(len(vars))
#             ]
#         )
#         return np.ravel(np.column_stack(intials))

#     def create_default_parameters(self, parameter_index: int, values) -> Array:
#         p_0 = self.cell_sim.create_problem().p[parameter_index]
#         return np.full(np.prod(self.shape), p_0)

#     def create_parameters(self, values) -> Array:
#         params = self.symbolic_cell.parameters
#         try:
#             intials = np.array(
#                 [
#                     np.ravel(
#                         values.get(
#                             params[i], self.create_default_parameters(i, values=values)
#                         ).astype(float, copy=False)
#                     )
#                     for i in range(len(params))
#                 ]
#             )
#             return np.ravel(np.column_stack(intials))
#         except ValueError:
#             return np.array([])

#     def create_func(self):
#         def func(t: float, y: Array, p: Array, dy: MutableArray):
#             for index in self.geometry.yield_locations():
#                 var_num = len(self.symbolic_cell.variables)
#                 param_num = len(self.symbolic_cell.parameters)
#                 one_d_index = self.geometry.get(index)
#                 one_d_slice = slice(var_num * one_d_index, var_num * (one_d_index + 1))
#                 parameter_slice = slice(
#                     param_num * one_d_index, param_num * (one_d_index + 1)
#                 )
#                 dy[one_d_slice] = self.cell_sim.compiled.func(
#                     t, y[one_d_slice], p[parameter_slice], dy[one_d_slice]
#                 )
#                 neighbors = self.geometry.iter_neighbors(index)
#                 for neighbor, interaction in neighbors:
#                     if interaction == "cell_interaction":
#                         neighbor_one_d_index = self.geometry.get(neighbor)
#                         neighbor_one_d_slice = slice(
#                             var_num * neighbor_one_d_index,
#                             var_num * (neighbor_one_d_index + 1),
#                         )
#                         neighbor_parameter_slice = slice(
#                             param_num * neighbor_one_d_index,
#                             param_num * (neighbor_one_d_index + 1),
#                         )
#                         dy[one_d_slice] += self.interact_sim.compiled.func(
#                             t,
#                             np.concatenate((y[one_d_slice], y[neighbor_one_d_slice])),
#                             np.concatenate(
#                                 (p[parameter_slice], p[neighbor_parameter_slice])
#                             ),
#                             np.zeros(2 * var_num, dtype=float),
#                         )[:var_num]
#                     if interaction == "boundary":
#                         dy[one_d_slice] += self.boundary_sim.compiled.func(
#                             t,
#                             y[one_d_slice],
#                             p[parameter_slice],
#                             np.zeros(var_num, dtype=float),
#                         )
#             return dy

#         return func

#     def create_problem(
#         self, values: Mapping[Variable | Parameter | Constant, ndarray], t_span
#     ):
#         y = self.create_initials(values=values)
#         p = self.create_parameters(values=values)
#         return Problem(
#             self.func,
#             t_span,
#             y,
#             p,
#             transform=lambda t, y, p, dy: y,
#             scale=np.ones_like(y),
#         )

#     def solve(
#         self,
#         *,
#         save_at: Sequence[Number],
#         values: Mapping[Variable | Parameter | Constant, ndarray] | None = None,
#         solver=LSODA(),
#         format: Literal["dataframe", "array"] = "dataframe",
#     ):
#         values = values if values is not None else {}
#         t_span = (save_at[0], save_at[-1])
#         problem = self.create_problem(values=values, t_span=t_span)
#         solution = solver(problem, save_at=np.asarray(save_at))
#         if len(self.symbolic_cell.variables) > 1:
#             output = solution.y.reshape(
#                 np.concatenate(
#                     [[len(save_at)], self.shape, [len(self.symbolic_cell.variables)]]
#                 )
#             )
#         else:
#             output = solution.y.reshape(np.concatenate([[len(save_at)], self.shape]))
#         if format == "dataframe":
#             cols = pd.MultiIndex.from_product(
#                 [range(s) for s in self.shape]
#                 + [[str(var) for var in self.symbolic_cell.variables]]
#             )
#             return pd.DataFrame(
#                 output.reshape(
#                     len(save_at),
#                     np.prod(self.shape) * len(self.symbolic_cell.variables),
#                 ),
#                 columns=cols,
#             )
#         if format == "array":
#             return output

#     def solve_and_animate(
#         self,
#         *,
#         save_at: Sequence[Number],
#         values: Mapping[Variable | Parameter | Constant, ndarray] | None = None,
#         solver=LSODA(),
#         variable: Variable,
#         timescale: Number = 1,
#         save_to: str | None = None,
#         **kwargs,
#     ):
#         if len(self.shape) != 2:
#             raise ValueError("only 2d automata can be animated")
#         full_result = self.solve(
#             save_at=save_at, values=values, solver=solver, format="array"
#         )
#         var_index = self.symbolic_cell.variables.index(variable)
#         var_num = len(self.symbolic_cell.variables)
#         result = full_result[..., var_index::var_num].reshape(
#             np.concatenate([[len(save_at)], self.shape])
#         )
#         interval = (save_at[1] - save_at[0]) * timescale * 1000
#         animation = animate_array(result, interval, **kwargs)
#         if isinstance(save_to, str):
#             animation.save(save_to, writer="pillow")
#         else:
#             return animation


# def animate_array(frames, interval=100, **kwargs):
#     """
#     frames: numpy array of shape (n_frames, H, W)
#     interval: delay between frames in ms
#     """
#     from matplotlib.animation import FuncAnimation
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()
#     im = ax.imshow(frames[0], animated=True, **kwargs)
#     ax.axis("off")
#     ax.set_frame_on(True)

#     def update(frame_idx):
#         im.set_array(frames[frame_idx])
#         return [im]

#     ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

#     return ani
