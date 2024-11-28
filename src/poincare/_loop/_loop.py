from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, ParamSpec, TypeVar

from ..types import System

P = ParamSpec("P")
T = TypeVar("T", bound=System)


@dataclass
class Loop(Generic[T]):
    system: T
