from collections.abc import Mapping

from ._node import Node
from .types import Initial, System


def get_from(param: str, model: type[System]) -> Node:
    path = param.split(".")
    current = model
    for level in path:
        current = getattr(current, level)
    return current


def to_values(
    params: Mapping[str, Initial], model: type[System]
) -> Mapping[Node, Initial]:
    return {
        get_from(param=param, model=model): value for param, value in params.items()
    }


__all__ = ["get_from", "to_values"]
