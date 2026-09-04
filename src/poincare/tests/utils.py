from typing import TypeVar

from .. import Derivative, Variable

T = TypeVar("T", Variable, Derivative)


def is_same_variable[T: (Variable, Derivative)](x: T, y: T, /) -> bool:
    return x == y


def is_derivative(derivative: Variable | Derivative, root: Variable) -> bool:
    while isinstance(derivative, Derivative):
        derivative = derivative.variable
    return derivative == root
