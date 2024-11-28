from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

import numba
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing_extensions import assert_never

from .. import System, Variable
from ..compile import (
    Compiled,
    build_first_order_symbolic_ode,
    build_first_order_vectorized_body,
    substitute,
)
from ..simulator import Problem, Simulator
from ..solvers import LSODA


@dataclass(repr=False)
class LoopSimulator:
    main: type[System]
    loop: type[System]
    main_variables_in_loop: Sequence[Variable]
    compiled_loop: Compiled[Variable, str] = field(init=False)

    def __post_init__(self):
        self.main_sim = Simulator(self.main)
        self.loop_sim = Simulator(self.loop)
        self.main_variables_in_loop = [
            x.variable if isinstance(x, Variable) else x
            for x in self.main_variables_in_loop
        ]
        self.compiled_loop = self._build_loop(self.main_sim.compiled)
        self.func = self._compile_func()
        self.func = numba.njit(self.func)

    def _build_loop(self, external: Compiled):
        symbolic = build_first_order_symbolic_ode(self.loop)

        to_exclude = set(self.main_variables_in_loop)
        variables = [v for v in symbolic.variables if v not in to_exclude]
        parameters = list(symbolic.parameters)

        mapping = {symbolic.independent[0]: "t"}
        for y_ext in self.main_variables_in_loop:
            ix = external.variables.index(y_ext)
            mapping[y_ext] = f"y[{ix}]"
        for i, y in enumerate(variables):
            mapping[y] = f"y[y_offset + {i}]"
        for i, p in enumerate(parameters):
            mapping[p] = f"p[p_offset + {i}]"

        diff_eqs = {k: substitute(v, mapping) for k, v in symbolic.func.items()}

        y_offset = len(external.variables)
        p_offset = len(external.parameters)
        y_step = len(variables)
        p_step = len(parameters)
        indent = 4 * " "
        lines = [
            f"N_loops = (y.size - {y_offset}) // {y_step}",
            "for loop_num in range(N_loops):",
            f"{indent}y_offset = loop_num * {y_step} + {y_offset}",
            f"{indent}p_offset = loop_num * {p_step} + {p_offset}",
        ]
        for k, eq in diff_eqs.items():
            ix = mapping.get(k).removeprefix("y")
            left = f"ydot{ix}"
            if k in self.main_variables_in_loop:
                lines.append(f"{indent}{left} += {eq}")
            else:
                lines.append(f"{indent}{left} = {eq}")
        loop = "\n".join(lines)
        return replace(symbolic, func=loop, variables=variables)

    def build_func(self):
        compiled = build_first_order_vectorized_body(self.main)
        loop = self._build_loop(compiled)
        return "\n    ".join(
            [
                compiled.func.removesuffix("return ydot"),
                *loop.func.splitlines(),
                "",
                "return ydot",
            ]
        )

    def _compile_func(self):
        func = self.build_func()
        lm = {}
        exec(func, globals(), lm)
        return lm["ode_step"]

    def create_initials(
        self,
        *,
        main_values: dict[Variable, float] = {},
        loop_values: dict[Variable, ArrayLike] = {},
    ):
        problem_main = self.main_sim.create_problem(values=main_values)

        y = [problem_main.y]
        p = [problem_main.p]
        mask = np.array(
            [
                i
                for i, x in enumerate(self.loop_sim.compiled.variables)
                if x in self.compiled_loop.variables
            ]
        )
        for _, values in pd.DataFrame(loop_values).iterrows():
            problem_loop = self.loop_sim.create_problem(values=values.to_dict())
            y.append(problem_loop.y[mask])
            p.append(problem_loop.p)
        y = np.concatenate(y)
        p = np.concatenate(p)
        return y, p

    def create_problem(
        self,
        *,
        main_values: dict[Variable, float] = {},
        loop_values: dict[Variable, ArrayLike] = {},
    ):
        y, p = self.create_initials(main_values=main_values, loop_values=loop_values)
        return Problem(
            self.func,
            (0, np.inf),
            y,
            p,
            transform=lambda t, y, p, dy: y,
            scale=np.ones_like(y),
        )

    def solve(
        self,
        *,
        main_values: dict[Variable, float] = {},
        loop_values: dict[Variable, ArrayLike] = {},
        solver=LSODA(),
        save_at: ArrayLike,
        loop_output: Literal["ignore", "sum", "index_as_suffix"] = "ignore",
    ):
        problem = self.create_problem(main_values=main_values, loop_values=loop_values)
        solution = solver(problem, save_at=np.asarray(save_at))
        main_variables = self.main_sim.compiled.variables
        if loop_output == "ignore":
            return pd.DataFrame(
                data=solution.y[:, : len(main_variables)],
                index=solution.t,
                columns=main_variables,
            )
        elif loop_output == "sum":
            loop_variables = self.compiled_loop.variables
            all_variables = [*main_variables, *loop_variables]
            y = solution.y[:, : len(all_variables)]
            y[:, len(main_variables) :] = (
                solution.y[:, len(main_variables) :]
                .reshape(solution.y.shape[0], -1, len(loop_variables))
                .sum(1)
            )
            return pd.DataFrame(y, index=solution.t, columns=all_variables)
        elif loop_output == "index_as_suffix":
            loop_variables = self.compiled_loop.variables
            all_variables = list(main_variables)
            for i in range(
                (solution.y.shape[1] - len(all_variables)) // len(loop_variables)
            ):
                all_variables.extend(f"{k}_{i}" for k in loop_variables)
            return pd.DataFrame(solution.y, index=solution.t, columns=all_variables)
        else:
            assert_never(loop_output)
