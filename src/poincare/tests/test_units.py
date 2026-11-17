import numpy as np
import pint
from pint import DimensionalityError, PintError
from pytest import mark, raises
from symbolite import real
from symbolite.impl import libstd
from symbolite.ops import translate

from poincare import (
    Constant,
    Derivative,
    Independent,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
    solvers,
)
from poincare.compile import Backend

u = pint.UnitRegistry()


@mark.parametrize(
    "value",
    [
        Independent(default=1 * u.s),
        Constant(default=1 * u.s),
        Parameter(default=1 * u.s),
        Variable(initial=1 * u.s),
    ],
)
def test_symbol_and_quantity(value):
    q = 1 * u.s

    left = value + q
    right = q + value
    assert translate(left, libstd) == translate(right, libstd)

    # TODO: what is the purpose of this test?
    left = left + q
    right = q + right
    assert translate(left, libstd) == translate(right, libstd)


def test_single_constant():
    class Model(System):
        x: Constant = assign(default=1 * u.m, constant=True)

    with raises(DimensionalityError):
        Model(x=1)
    with raises(DimensionalityError):
        Model(x=1 * u.m / u.s)


def test_single_parameter():
    class Model(System):
        x: Parameter = assign(default=1 * u.m)

    with raises(DimensionalityError):
        Model(x=1)
    with raises(DimensionalityError):
        Model(x=1 * u.m / u.s)


def test_single_variable():
    class Model(System):
        x: Variable = initial(default=1 * u.m)

    with raises(DimensionalityError):
        Model(x=1)
    with raises(DimensionalityError):
        Model(x=1 * u.m / u.s)


def test_explicit_unit_consistency():
    class Model(System):
        t = Independent(default=0 * u.s)
        x: Variable = initial(default=1)
        v: Derivative = x.derive(initial=1 / u.s)
        r: Parameter = assign(default=1 / u.s**2)

        eq = v.derive() << r

    with raises(PintError):

        class Model(System):
            t = Independent(default=0 * u.s)
            x: Variable = initial(default=1)
            v: Derivative = x.derive(initial=1 / u.m)
            r: Parameter = assign(default=1 / u.m**2)

            eq = v.derive() << r

    with raises(PintError):

        class Model(System):
            t = Independent(default=0 * u.s)
            x: Variable = initial(default=1)
            v: Derivative = x.derive(initial=1)
            r: Parameter = assign(default=1)

            eq = v.derive() << r


def test_implicit_unit_consistency():
    class Model(System):
        x: Variable = initial(default=1)
        v: Derivative = x.derive(initial=1 / u.m)
        r: Parameter = assign(default=1 / u.m**2)

        eq = v.derive() << r

    with raises(PintError):

        class Model(System):
            x: Variable = initial(default=1)
            v: Derivative = x.derive(initial=1 * u.m)
            r: Parameter = assign(default=1)

            eq = v.derive() << r

    with raises(PintError):

        class Model(System):
            x: Variable = initial(default=1)
            v: Derivative = x.derive(initial=1)
            r: Parameter = assign(default=1 * u.m)

            eq = v.derive() << r

    sim = Simulator(Model)
    with raises(PintError):
        sim.solve(save_at=list(range(3)) * u.s)
        sim = Simulator(Model)
    sim.solve(save_at=list(range(3)) * u.m)


def test_single_derivative():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=1 * u.m / u.s)

    with raises(DimensionalityError):
        Model(v=1)

    with raises(DimensionalityError):
        Model(v=1 * u.m)


def test_time():
    class Model(System):
        time = Independent(default=0 * u.s)
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << x / (time + 1 * u.s)


@mark.parametrize(
    "func",
    [
        lambda x: Independent(default=x),
        lambda x: Constant(default=x),
        lambda x: Parameter(default=x),
        lambda x: Variable(initial=x),
    ],
)
def test_dependencies(func):
    with raises(DimensionalityError):

        class WrongUnits(System):
            c0 = Constant(default=0 * u.s)
            c1 = func(c0 + 1)

    class Model(System):
        c0 = Constant(default=0 * u.s)
        c1 = func(c0 + 1 * u.s)


@mark.parametrize(
    "func",
    [
        lambda x: Independent(default=x),
        lambda x: Constant(default=x),
        lambda x: Parameter(default=x),
        lambda x: Variable(initial=x),
    ],
)
def test_eval_required(func):
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        y = func(None)
        eq = x.derive() << x * y
        # TODO: we don't know `y` units,
        # but we could infer what they must be,
        # and if they conflict somewhere else.


@mark.xfail(reason="Not yet implemented")
def test_required():
    class Model(System):
        x: Variable = initial()
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << -x / T

    with raises(DimensionalityError):
        Model(x=1 * u.m)

    Model(x=1)


def test_function():
    with raises(DimensionalityError):

        class WrongUnits(System):
            time = Independent(default=0 * u.s)
            p: Parameter = assign(default=real.cos(time))

    class Model(System):
        time = Independent(default=0 * u.s)
        p: Parameter = assign(default=real.cos(time * u.Hz))


def test_problem_units():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << -x / T

    sim = Simulator(Model)

    problem = sim.create_problem()
    assert problem.y == np.array([1])
    assert problem.p == np.array([1])
    assert problem.scale == [1 * u.m]

    problem = sim.with_values({Model.x: 10 * u.cm, Model.T: 10 * u.ms}).create_problem()
    assert problem.y == np.array([0.1])
    assert problem.p == np.array([0.01])
    assert problem.scale == [100 * u.cm]


def test_problem_with_transform_units():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << -x / T

    sim = Simulator(Model).with_transform({"x": Model.x * Model.T})

    problem = sim.create_problem()
    assert problem.y == np.array([1])
    assert problem.p == np.array([1])
    assert problem.scale == [1 * u.m * u.s]

    problem = sim.with_values({Model.x: 10 * u.cm, Model.T: 10 * u.ms}).create_problem()
    assert problem.y == np.array([0.1])
    assert problem.p == np.array([0.01])
    assert problem.scale == [100 * u.cm * 1000 * u.ms]


def test_simulator_values_and_save_at():
    class Model(System):
        x: Variable = initial(default=1)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << -x / T

    sim = Simulator(Model)

    with raises(DimensionalityError):
        sim.with_values({Model.x: 1 * u.m}).solve(save_at=list(range(3)) * u.s)

    with raises(DimensionalityError):
        sim.with_values({Model.T: 1}).solve(save_at=list(range(3)) * u.s)
    with raises(PintError):
        sim.solve(save_at=list(range(3)))

    sim.with_values({Model.T: 1 * u.ms}).solve(save_at=list(range(3)) * u.s)


def test_normalization():
    class Model(System):
        T: Parameter = assign(default=1 * u.s)
        x: Variable = initial(default=1 * u.m)
        y: Variable = initial(default=1 * u.m)
        eq_x = x.derive() << (x - y) / T
        eq_y = y.derive() << (x + y) / T

    t = np.linspace(0, 1, 10) * u.s
    sim = Simulator(Model)
    ds = sim.solve(save_at=t)
    ds_cm = sim.with_values({Model.y: 100 * u.cm}).solve(save_at=t)
    assert np.allclose((ds - ds_cm).pint.dequantify().to_dataarray(), 0)
    assert ds["y"].pint.units == u.m
    assert ds_cm["y"].pint.units == u.cm


LITERAL_SOLVER = solvers.LSODA(atol=1e-9, rtol=1e-9)


@mark.parametrize("backend", Backend.__args__)
@mark.parametrize(
    "rate",
    [1 / (1 * u.s), 1 / u.s, 1 / (1000 * u.ms), 60 / u.min],
    ids=["quantity", "unit", "ms", "min"],
)
def test_unit_in_equation(backend, rate):
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << -rate * x

    t = np.linspace(0, 2, 10) * u.s
    result = Simulator(Model, backend=backend).with_solver(LITERAL_SOLVER)
    x = result.solve(save_at=t)["x"]
    assert x.pint.units == u.m
    assert np.allclose(x.pint.magnitude, np.exp(-t.magnitude), rtol=1e-6)


def test_unit_in_equation_matches_parameter():
    class Literal(System):
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << -x / (2 * u.min)

    class WithParameter(System):
        x: Variable = initial(default=1 * u.m)
        tau: Parameter = assign(default=2 * u.min)
        eq = x.derive() << -x / tau

    t = np.linspace(0, 300, 10) * u.s
    literal = Simulator(Literal).with_solver(LITERAL_SOLVER).solve(save_at=t)
    parameter = Simulator(WithParameter).with_solver(LITERAL_SOLVER).solve(save_at=t)
    assert np.allclose(literal["x"].pint.magnitude, parameter["x"].pint.magnitude)


def test_quantity_as_equation():
    class Model(System):
        x: Variable = initial(default=0 * u.m)
        eq = x.derive() << 3 * u.km / u.hour

    t = np.array([0, 0.5, 1]) * u.hour
    x = Simulator(Model).with_solver(LITERAL_SOLVER).solve(save_at=t)["x"]
    assert np.allclose(x.pint.to(u.m).pint.magnitude, [0, 1500, 3000])


def test_unit_in_second_order_equation():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=0 * u.m / u.s)
        eq = v.derive() << -x / (1 * u.s) ** 2

    t = np.linspace(0, 5, 10) * u.s
    x = Simulator(Model).with_solver(LITERAL_SOLVER).solve(save_at=t)["x"]
    assert np.allclose(x.pint.magnitude, np.cos(t.magnitude), atol=1e-6)


def test_unit_in_time_dependent_parameter():
    class Model(System):
        time = Independent(default=0 * u.s)
        x: Variable = initial(default=0 * u.m)
        speed: Parameter = assign(default=2 * u.m / u.s**2 * time)
        eq = x.derive() << speed

    t = np.linspace(0, 3, 10) * u.s
    x = Simulator(Model).with_solver(LITERAL_SOLVER).solve(save_at=t)["x"]
    assert np.allclose(x.pint.magnitude, t.magnitude**2)


def test_unit_in_offset_literal():
    # Absolute temperatures are converted to kelvin, like the state.
    class Model(System):
        T: Variable = initial(default=u.Quantity(80, "degC").to("K"))
        k: Parameter = assign(default=0.1 / u.s)
        eq = T.derive() << -k * (T - u.Quantity(25, "degC"))

    t = np.linspace(0, 10, 5) * u.s
    T = Simulator(Model).with_solver(LITERAL_SOLVER).solve(save_at=t)["T"]
    expected = 273.15 + 25 + 55 * np.exp(-0.1 * t.magnitude)
    assert np.allclose(T.pint.magnitude, expected)


def test_unit_in_transform():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << -x / (1 * u.s)

    sim = Simulator(Model).with_transform({"y": Model.x + 1 * u.cm})
    y = sim.solve(save_at=np.array([0]) * u.s)["y"]
    assert y.pint.units == u.m
    assert np.allclose(y.pint.magnitude, 1.01)


def test_unit_in_equation_inconsistent():
    with raises(DimensionalityError):

        class Model(System):
            time = Independent(default=0 * u.s)
            x: Variable = initial(default=1 * u.m)
            eq = x.derive() << -x / (1 * u.s) + 1 * u.m


def test_unit_in_equation_implicit_time():
    # Without an explicit Independent, the literal defines the time units.
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << -(1 * u.m / u.s) * x

    sim = Simulator(Model)
    with raises(PintError):
        sim.solve(save_at=np.linspace(0, 1, 3) * u.s)
    sim.solve(save_at=np.linspace(0, 1, 3) * u.s / u.m)


def test_zero_initial_with_unit():
    class Model(System):
        x: Variable = initial(default=0 * u.m)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << x / T

    times = np.linspace(0, 1, 10) * u.s
    sim = Simulator(Model)
    sim.solve(save_at=times)


def test_v_derive():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=0 * u.m / u.s)
        w: Parameter = assign(default=1 * u.Hz)
        eq = v.derive() << -(w**2) * x
