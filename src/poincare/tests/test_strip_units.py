import numpy as np
import pint
from pytest import approx
from symbolite.impl import libstd
from symbolite.ops import translate

from poincare import (
    Derivative,
    Independent,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
    strip_units,
)
from poincare.reactions.reactions import MassAction, RateLaw
from poincare.types import Constant, Equation, EquationGroup

u = pint.UnitRegistry()


def assert_unitless(model: type[System]):
    for node in model._yield(Independent | Constant | Parameter | Variable):
        assert not isinstance(translate(node, libstd), pint.Quantity), node
    for node in model._yield(Equation | EquationGroup):
        for eq in getattr(node, "equations", [node]):
            assert not isinstance(translate(eq.rhs, libstd), pint.Quantity), eq


def solve(model, save_at):
    return Simulator(model).solve(save_at=save_at)


class Decay(System):
    t = Independent(default=0 * u.s)
    x: Variable = initial(default=1 * u.mM)
    k: Parameter = assign(default=2 / u.min)
    s: Parameter = assign(default=3 * u.uM / u.s)
    eq = x.derive() << -k * x + s


def test_magnitudes_in_base_units():
    Unitless = strip_units(Decay)

    assert Unitless.__name__ == "Decay"
    assert_unitless(Unitless)
    assert Unitless.x.initial == approx(1)
    assert Unitless.k.default == approx(2 / 60)
    assert Unitless.s.default == approx(3e-3)
    assert Unitless.eq == (
        Unitless.x.derive() << -Unitless.k * Unitless.x + Unitless.s
    )


def test_units_dict():
    Unitless = strip_units(Decay)

    # keys are the components of the new system, not the original
    keys = {str(k): k for k in Unitless._units}
    assert keys["x"] is Unitless.x
    assert keys["t"] is Unitless.t

    assert Unitless._units == {
        Unitless.t: u.s,
        Unitless.x: u.mol / u.m**3,
        Unitless.k: 1 / u.s,
        Unitless.s: u.mol / u.m**3 / u.s,
    }


def test_simulation_matches():
    Unitless = strip_units(Decay)
    t = np.linspace(0, 100, 5)

    expected = solve(Decay, t * u.s).x.pint.to(Unitless._units[Unitless.x]).values
    result = solve(Unitless, t).x.values
    assert result == approx(expected)


def test_inline_quantity():
    class Model(System):
        x: Variable = initial(default=1 * u.mM)
        eq = x.derive() << -x / (2 * u.min) + 3 * u.uM / u.s

    Unitless = strip_units(Model)
    assert_unitless(Unitless)
    assert Unitless.eq.lhs == Unitless.x.derive()
    assert translate(Unitless.eq.rhs, libstd) == approx(-1 / 120 + 3e-3)
    # The independent is not declared, so its units are stored as "time"
    assert Unitless._units == {Unitless.x: u.mol / u.m**3, "time": u.s}


def test_derivatives():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=0 * u.km / u.h)
        k: Parameter = assign(default=1 / u.s**2)
        eq = v.derive() << -k * x

    Unitless = strip_units(Model)
    assert_unitless(Unitless)
    assert Unitless.v is Unitless.x.derivatives[1]
    assert Unitless.x.equation_order == 2
    assert Unitless._units == {
        Unitless.x: u.m,
        Unitless.v: u.m / u.s,
        Unitless.k: 1 / u.s**2,
        "time": u.s,
    }

    t = np.linspace(0, 5, 5)
    expected = solve(Model, t * u.s).x.pint.to(u.m).values
    assert solve(Unitless, t).x.values == approx(expected)


def test_subsystem():
    class Sub(System):
        x: Variable = initial(default=1 * u.mM)
        k: Parameter = assign(default=1 / u.min)
        eq = x.derive() << -k * x

    class Model(System):
        t = Independent(default=0 * u.s)
        y: Variable = initial(default=2 * u.mM)
        sub = Sub(x=y, k=2 / u.min)
        z: Variable = initial(default=0 * u.mM)
        eq = z.derive() << sub.k * sub.x

    Unitless = strip_units(Model)
    assert_unitless(Unitless)
    assert Unitless.sub.x is Unitless.y
    assert Unitless.sub.k.default == approx(2 / 60)
    assert Unitless.eq == (
        Unitless.z.derive() << Unitless.sub.k * Unitless.sub.x
    )
    assert Unitless._units[Unitless.sub.k] == 1 / u.s
    assert {str(k): k for k in Unitless._units}["sub.k"] is Unitless.sub.k
    assert Unitless._units[Unitless.z] == u.mol / u.m**3

    t = np.linspace(0, 100, 5)
    expected = solve(Model, t * u.s)
    result = solve(Unitless, t)
    for name in ("y", "z"):
        assert result[name].values == approx(
            expected[name].pint.to(Unitless._units[getattr(Unitless, name)]).values
        )


def test_instance():
    class Model(System):
        x: Variable = initial(default=1 * u.mM)
        k: Parameter = assign(default=1 / u.min)
        eq = x.derive() << -k * x

    unitless = strip_units(Model(k=1 / u.s))
    assert isinstance(unitless, System)
    assert unitless.k.default == 1
    assert unitless._units[unitless.k] == 1 / u.s


def test_reactions():
    class Model(System):
        t = Independent(default=0 * u.s)
        x: Variable = initial(default=1 * u.mM)
        y: Variable = initial(default=0 * u.mM)
        k: Parameter = assign(default=1 / u.min)
        r1 = MassAction(reactants=[x], products=[y], rate=k)
        r2 = RateLaw(reactants=[y], products=[], rate_law=1 * u.uM / u.s)

    Unitless = strip_units(Model)
    assert_unitless(Unitless)
    assert Unitless._r2_rate_law.default == approx(1e-3)

    t = np.linspace(0, 100, 5)
    expected = solve(Model, t * u.s)
    result = solve(Unitless, t)
    for name in ("x", "y"):
        assert result[name].values == approx(
            expected[name].pint.to(Unitless._units[getattr(Unitless, name)]).values
        )


def test_without_units():
    class Model(System):
        x: Variable = initial(default=1)
        k: Parameter = assign(default=2)
        eq = x.derive() << -k * x

    Unitless = strip_units(Model)
    assert Unitless.x.initial == 1
    assert Unitless.k.default == 2
    assert Unitless.eq == (Unitless.x.derive() << -Unitless.k * Unitless.x)
    assert Unitless._units == {Unitless.x: None, Unitless.k: None, "time": None}


def test_no_units_differs_from_dimensionless():
    class Model(System):
        t = Independent(default=0 * u.s)
        x: Variable = initial(default=1 * u.mM)
        n: Parameter = assign(default=2)
        ratio: Parameter = assign(default=1 * u.m / u.km)
        eq = x.derive() << -n * ratio * x / u.s

    Unitless = strip_units(Model)
    assert Unitless._units[Unitless.n] is None
    assert Unitless._units[Unitless.ratio] == u.dimensionless
    assert Unitless.ratio.default == approx(1e-3)

