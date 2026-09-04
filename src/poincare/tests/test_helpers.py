import pytest

from ..helpers import get_from, to_values
from ..types import Derivative, Parameter, System, Variable, assign, initial


class Oscillator(System):
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)

    omega: Parameter = assign(default=1)
    k: Parameter = assign(default=2)

    spring = vx.derive() << -(omega**2) * x


class Parent(System):
    osc = Oscillator()


class ParentParent(System):
    subparent = Parent()


def test_get_from():
    x_str = str(ParentParent.subparent.osc.x)
    assert x_str == "subparent.osc.x"
    assert get_from(x_str, ParentParent) == ParentParent.subparent.osc.x
    with pytest.raises(AttributeError):
        get_from("noexistent", ParentParent)


def test_to_values():
    param_dict = to_values({"subparent.osc.x": 1}, ParentParent)
    assert param_dict == {ParentParent.subparent.osc.x: 1}
    with pytest.raises(AttributeError):
        to_values({"noexistent": 1}, ParentParent)
