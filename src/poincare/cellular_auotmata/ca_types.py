from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..types import System
from numpy import ndarray
from ..types import (
    Initial,
    Variable,
    Parameter,
    Derivative,
    Constant,
    Independent,
    System,
)
from typing import Callable, Protocol, Generator, Iterable, Mapping, Sequence, TypeVar

import numpy as np
from symbolite import Symbol

type VariableContainer = type[System] | SystemWrapper
LocationVariable = TypeVar("LocationVariable")
type Location[LocationVariable] = LocationVariable
type Index = int
type InteractionType = str


@dataclass
class Automata:
    cells: type[System]
    geometry: Geometry
    interactions: Mapping[InteractionType, Interaction]

    def __post_init__(self):
        self.interaction_systems = {
            interaction: create_interaction_system(
                interaction, Geometry.interaction_types[interaction]
            )
            for interaction in interactions.keys()
        }

    # interact: Callable[[type, type[System] | SystemWrapper, type[System] | SystemWrapper], None]
    # boundary: Callable[[type, type[System]], None]


type Interaction = Callable[
    [type, type[System] | SystemWrapper, type[System] | SystemWrapper], None
]

type Boundary = Callable[[type, type[System]], None]


class Geometry(Protocol):
    def __init__(self, shape): ...
    def iter_neighbors(
        self, cell: Location
    ) -> Iterable[tuple[Location, InteractionType]]: ...

    def get(self, location: Location) -> Index: ...

    def yield_locations(self) -> Iterable[Location]: ...


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
        external_wrapper = SystemWrapper(wrappers["out"])
        cls.interaction(internal_wrapper, external_wrapper)
        return super().__init_subclass__()


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
