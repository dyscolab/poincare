import pint
from pytest import approx, raises

from ..simulator import Simulator
from ..types import Independent, System, Variable, initial
from ..units import strip_units

u = pint.UnitRegistry()


def test_derivative_outside_class():
    class Base(System):
        x: Variable = initial(default=0)

    Base.x.derive(initial=0)
    assert 1 not in Base.x.derivatives


def test_derivative_in_subclass():
    class Base(System):
        x: Variable = initial(default=0)

    with raises(TypeError):

        class Extended(Base):
            vx = Base.x.derive(initial=0)

    assert 1 not in Base.x.derivatives


def test_equation_outside_class():
    class Base(System):
        x: Variable = initial(default=0)

    Base.x.derive() << 1
    assert Base.x.equation_order is None


def test_equation_in_subclass():
    class Base(System):
        x: Variable = initial(default=0)

    with raises(TypeError):

        class Extended(Base):
            eq = Base.x.derive() << 1

    assert Base.x.equation_order is None


def test_subclass_concrete_system():
    class Model(System):
        x: Variable = initial(default=0)

    with raises(TypeError):

        class Extended(Model):
            pass

    with raises(TypeError):

        class AbstractExtended(Model, abstract=True):
            pass


def test_subclass_concrete_system_with_abstract_base():
    class Base(System, abstract=True):
        pass

    class Model(Base):
        x: Variable = initial(default=0)

    with raises(TypeError):

        class Extended(Model):
            pass


def test_inherit_from_abstract():
    class Base(System, abstract=True):
        t = Independent(default=0 * u.s)
        y: Variable = initial(default=2)

    class Model(Base):
        x: Variable = initial(default=1)
        eq = x.derive() << -x / (1 * u.s)

    assert set(Model._yield(Variable)) == {Model.x, Model.y}
    assert list(Model._yield(Independent)) == [Model.t]
    assert Model.y is Base.y
    assert Model(y=3).y.initial == 3

    compiled = Simulator(strip_units(Model)).compiled
    assert [str(x) for x in compiled.independent] == ["t"]


def test_strip_units_inherited():
    class Base(System, abstract=True):
        t = Independent(default=0 * u.s)
        y: Variable = initial(default=2 * u.mM)

    class Model(Base):
        x: Variable = initial(default=1 * u.mM)
        eq = x.derive() << -x / (1 * u.min) + Base.y / (1 * u.min)

    Unitless = strip_units(Model)
    assert Unitless.y.initial == approx(2)
    assert Unitless.eq == (
        Unitless.x.derive() << -Unitless.x / 60 + Unitless.y / 60
    )
    assert Unitless._units[Unitless.t] == u.s
    assert Unitless._units[Unitless.y] == u.mol / u.m**3
