from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..types import System
from ..types import (
    Initial,
    Variable,
    Parameter,
    Derivative,
    Constant,
    Independent,
    System,
)
from ..simulator import Simulator, Components
from typing import Callable, Protocol, Generator, Iterable, Mapping, Sequence, TypeVar
from ..compile import build_first_order_symbolic_ode, Backend, Array, ExprRHS, Compiled

import numpy as np
from symbolite import Real, Symbol
from symbolite import real

type VariableContainer = type[System] | SystemWrapper
LocationVariable = TypeVar("LocationVariable")
type Location[LocationVariable] = LocationVariable
ArbitraryIntialVariable = TypeVar("ArbitraryIntialVariable")
type ArbitraryInitial[ArbitraryIntialVariable] = ArbitraryIntialVariable
type Index = int
type InteractionType = str
type CellType = str


@dataclass
class Automata:
    cells: type[System] | Sequence[type[System]] | Geometry.Cells
    geometry: type[Geometry]
    interactions: Geometry.Interactions
    backend: Backend = "numpy"
    def __post_init__(self):
        if isinstance(self.cells, type[System] | Sequence[type[System]]):
                wrapped_cells = self.geometry.Cells(self.cells)
        else: 
            wrapped_cells = self.cells
        if isinstance(self.interactions, Callable | Sequence[Callable]):
                wrapped_interactions = self.geometry.Interactions(self.interactions)
        else: 
            wrapped_interactions = self.interactions
        self.cell_systems = {cell_category: [cell for cell in wrapped_cells.__getattribute__(cell_category)] for cell_category in self.geometry.Cells.cell_categories}
        self.interaction_systems = {interaction_category: [builder(interaction) for interaction in wrap_with_iterable(wrapped_interactions.__getattribute__(interaction_category))] for interaction_category, builder in self.geometry.Interactions.builders.items()}
 
        self.symbolic_cells = {
            cell_category: [build_first_order_symbolic_ode(cell) for cell in cells] for cell_category, cells in self.cell_systems.items()
        }
        self.symbolic_interactions = {
            interaction_category: [build_first_order_symbolic_ode(interaction) for interaction in interactions]
            for interaction_category, interactions in self.interaction_systems.items()
        }

        self.cell_sims = {
            cell_category: [Simulator(cell) for cell in cells]
            for cell_category, cells in self.cell_systems.items()
        }        
        self.interaction_sims = {
            interaction_category: [Simulator(interaction) for interaction in interactions]
            for interaction_category, interactions in self.interaction_systems.items()
        }

type Interaction = Callable[
    [type, type[System] | SystemWrapper, type[System] | SystemWrapper], None
]

type Boundary = Callable[[type, type[System]], None]


class Geometry(Protocol):
    cell_counts: Mapping[CellType, Sequence[int]]

    def __init__(self, shape): ... 

    def iter_neighbors(
        self, cell: Location
    ) -> Iterable[tuple[Location, InteractionType]]: ...

    def get(self, location: Location) -> Index: ...

    def yield_locations(self) -> Iterable[tuple[Location, CellType, Index]]: ...

    def calculate_cell_counts(self) -> Mapping[CellType, Sequence[int]]: ...
    
    def create_initials(self, symbolic_cells: Mapping[CellType, Sequence[Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]]], cell_sims: Mapping[CellType, Sequence[Simulator]], values: Mapping[CellType, Sequence[Mapping[Components | real.Real, ArbitraryInitial]]]) -> Array: ...
    
    def create_parameters(self, symbolic_cells: Mapping[CellType, Sequence[Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]]], cell_sims: Mapping[CellType, Sequence[Simulator]], values: Mapping[CellType, Sequence[Mapping[Components | real.Real, ArbitraryInitial]]]) -> Array: ...
    
    class Interactions:
        builders: Mapping[InteractionType, Callable]
        def __init__(self, *args) -> None: ...
        
    class Cells:
        cell_categories: Sequence[CellType]     
        def __init__(self, *args) -> None: ... 
           
class BaseGeometry:
    def __init__(self, shape): 
        self.cell_counts = self.calculate_cell_counts()

    def iter_neighbors(
        self, cell: Location
    ) -> Iterable[tuple[Location, InteractionType]]: ...

    def get(self, location: Location) -> Index: ...

    def yield_locations(self) -> Iterable[tuple[Location, CellType, Index]]: ...

    class Interactions:
        builders: Mapping[InteractionType, Callable]
        def __init__(self, *args) -> None: ...
        
    class Cells:
        cell_categories: Sequence[CellType]     
        def __init__(self, *args) -> None: ... 

    def calculate_cell_counts(self) -> Mapping[CellType, Sequence[int]]:
        counts = {cell_category: [0] for cell_category in self.Cells.cell_categories}
        for loc, category, index in self.yield_locations():
            try:
                counts[category][index] +=1
            except IndexError:
                counts[category] += [0] * (index - len(category) + 1)
        return counts
    
    # TODO: initials set from constant won't respond to changes in constants from values 
    def create_default_initials(self, cell_category: CellType, index: Index, cell_sims: Mapping[CellType, Sequence[Simulator]], variable_index: int) -> Array:
        y_0 = cell_sims[cell_category][index].create_problem().y[variable_index]
        return np.full(self.cell_counts[cell_category][index], y_0)
        
    
    def create_cell_initials(self, cell_category: CellType, index: Index, symbolic_cells: Mapping[CellType, Sequence[Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]]], cell_sims: Mapping[CellType, Sequence[Simulator]], values: Mapping[CellType, Sequence[Mapping[Components | real.Real, ArbitraryInitial[Array]]]]) -> Array:
        vars = symbolic_cells[cell_category][index].variables
        intials = [
                np.ravel(np.asarray(
                    values[cell_category][index].get(
                        vars[i],
                        self.create_default_initials(cell_category=cell_category, index=index, cell_sims=cell_sims, variable_index=i),
                    ), dtype=float)
                )
                for i in range(len(vars))
        ]
        return np.ravel(np.column_stack(intials))

    def create_initials(self, symbolic_cells: Mapping[CellType, Sequence[Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]]], cell_sims: Mapping[CellType, Sequence[Simulator]], values: Mapping[CellType, Sequence[Mapping[Components | real.Real, ArbitraryInitial[Array]]]]) -> Array:
        initials = []
        for cell_category, count_list in self.cell_counts.items():
            for index in range(len(count_list)):
                initials.append(self.create_cell_initials(cell_category=cell_category, index=index, symbolic_cells=symbolic_cells, cell_sims=cell_sims, values=values))
            
        return np.concatenate(initials)

    def create_default_parameters(self, cell_category: CellType, index: Index, cell_sims: Mapping[CellType, Sequence[Simulator]], parameter_index: int) -> Array:
        p_0 = cell_sims[cell_category][index].create_problem().p[parameter_index]
        return np.full(self.cell_counts[cell_category][index], p_0)
        
    
    def create_cell_parameters(self, cell_category: CellType, index: Index, symbolic_cells: Mapping[CellType, Sequence[Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]]], cell_sims: Mapping[CellType, Sequence[Simulator]], values: Mapping[CellType, Sequence[Mapping[Components | real.Real, ArbitraryInitial[Array]]]]) -> Array:
        params = symbolic_cells[cell_category][index].parameters
        intials = [
                np.ravel(np.asarray(
                    values[cell_category][index].get(
                        params[i],
                        self.create_default_parameters(cell_category=cell_category, index=index, cell_sims=cell_sims, parameter_index=i),
                    ), dtype=float)
                )
                for i in range(len(params))
        ]
        return np.ravel(np.column_stack(intials))

    def create_parameters(self, symbolic_cells: Mapping[CellType, Sequence[Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]]], cell_sims: Mapping[CellType, Sequence[Simulator]], values: Mapping[CellType, Sequence[Mapping[Components | real.Real, ArbitraryInitial[Array]]]]) -> Array:
        initials = []
        for cell_category, count_list in self.cell_counts.items():
            for index in range(len(count_list)):
                initials.append(self.create_cell_parameters(cell_category=cell_category, index=index, symbolic_cells=symbolic_cells, cell_sims=cell_sims, values=values))
            
        return np.concatenate(initials)
    
def wrap_with_iterable(a: object) -> Iterable: 
    if isinstance(a, Iterable):
        return a
    else:
        yield a
class Square(Geometry):
    interaction_types = {"cell_interaction": 2, "boundary": 1}

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = np.asarray(shape)
        self.dimension = len(shape)
        self.strides = calculate_strides(self.shape, self.dimension)

    def yield_locations(
        self,
    ) -> Iterable[Location[Sequence[int]]]:
        return product(*[range(i) for i in self.shape])

    def iter_neighbors(
        self, cell: Location
    ) -> Iterable[tuple[Location[Sequence[int]], InteractionType]]:
        cell = np.asarray(cell)
        for i in range(self.dimension):
            for j in [-1, 1]:
                index = np.array([j if i == k else 0 for k in range(self.dimension)])
                target = cell + index
                if np.all(0 <= target) and np.all(target <= self.shape - 1):
                    yield target, "cell_interaction"
                else:
                    yield cell, "boundary"

    def get(self, location: Location) -> Index:
        strides = calculate_strides(self.shape, len(self.shape))
        return int(np.sum(strides * location))

    class Cells:
        def __init__(self):
            self.cell_categories: Sequence[Sequence[str]]

    class Interactions:
        def __init__(self, interaction: Callable[[type[System], VariableContainer, VariableContainer], None], boundary: Callable[[type[System], VariableContainer], None] ):
            self.interaction = interaction
            self.boundary= boundary
            self.builders = {"interaction": }


def calculate_strides(shape, dim):
    return np.array([np.prod(shape[k + 1 : dim]) for k in range(dim)])


@dataclass
class SquareAutomata:
    cell: type[System]
    cell_interaction: Interaction
    boundary: Callable[[type, type[System]], None]
    geometry: type[Geometry] = Square

    def __post_init__(self):
        self.interaction_system = create_cell_interaction(
            self.cell_interaction, self.cell
        )
        self.boundary_system = create_boundary(self.boundary, self.cell)


@dataclass
class SystemWrapper:
    attributes: Mapping[str, Variable | Parameter | Constant | Independent]

    def __post_init__(self):
        for key, value in self.attributes.items():
            setattr(self, key, value)


class CellInteraction(System, abstract=True):
    @classmethod
    def interaction(cls, int, ext):
        pass

    def __init_subclass__(cls, cell: type[System]) -> None:
        wrappers = {"int": {}, "out": {}}
        for name in wrappers.keys():
            for attribute in cell._yield(Variable | Parameter | Constant | Independent):
                if isinstance(attribute, Variable):
                    wrappers[name][str(attribute)] = attribute._copy_from(None)
                    wrappers[name][str(attribute)].__set_name__(
                        cls, name + "_" + str(attribute)
                    )
                    setattr(
                        cls, name + "_" + str(attribute), wrappers[name][str(attribute)]
                    )
                    for order, der in wrappers[name][
                        str(attribute)
                    ].derivatives.items():
                        wrappers[name][attribute.derivatives[order].name] = der
                        wrappers[name][attribute.derivatives[order].name].__set_name__(
                            cls, name + "_" + attribute.derivatives[order].name
                        )
                        setattr(
                            cls,
                            name + "_" + attribute.derivatives[order].name,
                            wrappers[name][attribute.derivatives[order].name],
                        )
                    if name == "out":
                        try:
                            max_order = max(
                                wrappers[name][str(attribute)].derivatives.keys()
                            )
                            setattr(
                                cls,
                                "out_eq" + str(attribute),
                                wrappers[name][str(attribute)]
                                .derivatives[max_order]
                                .derive()
                                << 0,
                            )
                        except ValueError:
                            setattr(
                                cls,
                                "out_eq" + str(attribute),
                                wrappers[name][str(attribute)].derive() << 0,
                            )
                elif isinstance(attribute, Constant):
                    wrappers[name][str(attribute)] = attribute._copy_from(None)
                    wrappers[name][str(attribute)].__set_name__(
                        cls, name + "_" + str(attribute)
                    )
                    setattr(
                        cls, name + "_" + str(attribute), wrappers[name][str(attribute)]
                    )
                elif isinstance(attribute, Independent):
                    if name == "int":
                        wrappers[name][str(attribute)] = attribute._copy_from(None)
                        wrappers[name][str(attribute)].__set_name__(cls, str(attribute))
                        setattr(cls, str(attribute), wrappers[name][str(attribute)])
                elif isinstance(attribute, Parameter):
                    if isinstance(attribute.default, Real):
                        wrappers[name][str(attribute)] = Parameter(
                            default=attribute.default.subs(
                                {
                                    cell.__dict__[attr]: wrappers[name][attr]
                                    for attr in wrappers[name].keys()
                                }
                            )
                        )
                        wrappers[name][str(attribute)].__set_name__(
                            cls, name + "_" + str(attribute)
                        )
                        setattr(
                            cls,
                            name + "_" + str(attribute),
                            wrappers[name][str(attribute)],
                        )
                    else:
                        wrappers[name][str(attribute)] = attribute._copy_from(None)
                        wrappers[name][str(attribute)].__set_name__(
                            cls, name + "_" + str(attribute)
                        )
                        setattr(
                            cls,
                            name + "_" + str(attribute),
                            wrappers[name][str(attribute)],
                        )
                else:
                    wrappers[name][str(attribute)] = attribute

        internal_wrapper = SystemWrapper(wrappers["int"])
        external_wrapper = SystemWrapper(wrappers["out"])
        cls.interaction(internal_wrapper, external_wrapper)
        return super().__init_subclass__()


# TODO: modify so it can take different internal and external cells, change name to build_binary_interaction
def create_cell_interaction(
    interaction: Interaction, cell: type[System]
) -> type[System]:
    class InteractionSystem(CellInteraction, cell=cell):
        @classmethod
        def interaction(cls, int, ext):
            interaction(cls, int, ext)

    return InteractionSystem


class BoundaryCreator(System, abstract=True):
    @classmethod
    def interaction(cls, int):
        pass

    def __init_subclass__(cls, cell: type[System]) -> None:
        wrappers = {"int": {}}
        for name in wrappers.keys():
            for attribute in cell._yield(Variable | Parameter | Constant | Independent):
                if isinstance(attribute, Variable):
                    wrappers[name][str(attribute)] = attribute._copy_from(None)
                    wrappers[name][str(attribute)].__set_name__(
                        cls, name + "_" + str(attribute)
                    )
                    setattr(
                        cls, name + "_" + str(attribute), wrappers[name][str(attribute)]
                    )
                    for order, der in wrappers[name][
                        str(attribute)
                    ].derivatives.items():
                        wrappers[name][attribute.derivatives[order].name] = der
                        wrappers[name][attribute.derivatives[order].name].__set_name__(
                            cls, name + "_" + attribute.derivatives[order].name
                        )
                        setattr(
                            cls,
                            name + "_" + attribute.derivatives[order].name,
                            wrappers[name][attribute.derivatives[order].name],
                        )
                elif isinstance(attribute, Constant):
                    wrappers[name][str(attribute)] = attribute._copy_from(None)
                    wrappers[name][str(attribute)].__set_name__(
                        cls, name + "_" + str(attribute)
                    )
                    setattr(
                        cls, name + "_" + str(attribute), wrappers[name][str(attribute)]
                    )
                elif isinstance(attribute, Independent):
                    wrappers[name][str(attribute)] = attribute._copy_from(None)
                    wrappers[name][str(attribute)].__set_name__(cls, str(attribute))
                    setattr(cls, str(attribute), wrappers[name][str(attribute)])
                elif isinstance(attribute, Parameter):
                    if isinstance(attribute.default, Symbol):
                        wrappers[name][str(attribute)] = Parameter(
                            default=attribute.default.subs(
                                {
                                    cell.__dict__[attr]: wrappers[name][attr]
                                    for attr in wrappers[name].keys()
                                }
                            )
                        )
                        wrappers[name][str(attribute)].__set_name__(
                            cls, name + "_" + str(attribute)
                        )
                        setattr(
                            cls,
                            name + "_" + str(attribute),
                            wrappers[name][str(attribute)],
                        )
                    else:
                        wrappers[name][str(attribute)] = attribute._copy_from(None)
                        wrappers[name][str(attribute)].__set_name__(
                            cls, name + "_" + str(attribute)
                        )
                        setattr(
                            cls,
                            name + "_" + str(attribute),
                            wrappers[name][str(attribute)],
                        )
                else:
                    wrappers[name][str(attribute)] = attribute

        internal_wrapper = SystemWrapper(wrappers["int"])
        cls.boundary(internal_wrapper)
        return super().__init_subclass__()


def create_boundary(boundary: Boundary, cell: type[System]) -> type[System]:
    class BoundarySystem(BoundaryCreator, cell=cell):
        @classmethod
        def boundary(cls, int):
            boundary(cls, int)

    return BoundarySystem


class AnisotropicSquare:
    pass
