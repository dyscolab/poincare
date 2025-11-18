from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping, Hashable

from symbolite import Symbol

from .ca_types import Automata
from ..types import System
from ..compile import Backend, build_first_order_symbolic_ode, Array, MutableArray
from ..simulator import Simulator 
from ..types import Variable, Derivative, Number, Initial
import numpy as np
from numpy import ndarray


@dataclass
class CASimulator:
    model: Automata
    shape: tuple[int]
    backend: Backend = "numpy"
    transform: Sequence[Symbol] | Mapping[Hashable, Symbol] | None = None

    def __post_init__(self):
        self.cell_sim = Simulator(self.model.cell)
        self.interact_sim = Simulator(self.model.interact)
        self.symbolic_cell = build_first_order_symbolic_ode(self.model.cell)
        self.symbolic_interact = build_first_order_symbolic_ode(self.model.interact)

    def create_default_initials(self, variable: Variable | Derivative) -> Array:
        y_0 = self.cell_sim.create_problem().y
        return np.tile(y_0, np.prod(self.shape))
    def create_initials(self, values) -> Array:
        intials = np.array([np.ravel(values.get(var, self.create_default_initials(var)))  for var in self.symbolic_cell.variables])
        return np.ravel(np.column_stack(intials))
    
    def create_default_parameters(self, variable: Symbol) -> Array:
        p_0 = self.cell_sim.create_problem().p
        return np.tile(p_0, np.prod(self.shape))
    def create_parameters(self, values) -> Array:
        intials = np.array([np.ravel(values.get(param, self.create_default_parameters(param))) for param in self.symbolic_cell.parameters])
        return np.ravel(np.column_stack(intials))
    
    def create_func(self):
        new_shape = np.concat([self.shape, (len(self.symbolic_cell.variables),)])
        def func(t: float, y: Array, p: Array, dy: MutableArray):  
            system = np.reshape(y, shape=new_shape)
            

    def create_problem(self):
        own = self.cell_sim.compiled.func

    def solve(self, save_at: Sequence[Number], values: Mapping[Variable, ndarray]):



    
