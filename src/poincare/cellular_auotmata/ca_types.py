from __future__ import annotations

from dataclasses import dataclass

from ..types import System
from numpy import ndarray
from ..types import Initial
from typing import Callable, Protocol, Generator

class SystemWrapperMeta(type):
    pass
class SystemWrapper(metaclass = SystemWrapperMeta):


@dataclass
class Automata:
    cell: type[System]
    interact: Callable[[type, type[System] | SystemWrapper, type[System] | SystemWrapper], None]
    boundary: Callable[[type, type[System]], None]


type Interaction = Callable[[type, type[System] | SystemWrapper, type[System] | SystemWrapper], None]


class Connections(Protocol):

    def iter_neighbors(self, cell) -> Generator[tuple[System, System, Interaction]]:
        ...


class Square:

    def __init__(self) -> None:

    def iter_neighbors(self):
        pass


class AnisotropicSquare
