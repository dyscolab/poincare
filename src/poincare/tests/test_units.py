import numpy as np
from pint import DimensionalityError, get_application_registry
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
)

u = get_application_registry()


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


def test_single_derivative():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=1 * u.m / u.s)

    with raises(DimensionalityError):
        Model(v=1)

    with raises(DimensionalityError):
        Model(v=1 * u.m)

    with raises(DimensionalityError):

        class NoUnits(System):
            x: Variable = initial(default=1 * u.m)
            v: Derivative = x.derive(initial=1)

    with raises(DimensionalityError):

        class WrongUnits(System):
            x: Variable = initial(default=1 * u.m)
            v: Derivative = x.derive(initial=1 * u.m)


def test_single_equation():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << 1 * u.m / u.s

    with raises(DimensionalityError):

        class NoUnits(System):
            x: Variable = initial(default=1 * u.m)
            eq = x.derive() << 1

    with raises(DimensionalityError):

        class WrongUnits(System):
            x: Variable = initial(default=1 * u.m)
            eq = x.derive() << 1 * u.m

    with raises(DimensionalityError):

        class WrongVariableUnits(System):
            x: Variable = initial(default=1 * u.m)
            eq = x.derive() << x


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

    problem = sim.create_problem(values={Model.x: 10 * u.cm, Model.T: 10 * u.ms})
    assert problem.y == np.array([0.1])
    assert problem.p == np.array([0.01])
    assert problem.scale == [100 * u.cm]


def test_problem_with_transform_units():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << -x / T

    sim = Simulator(Model, transform={"x": Model.x * Model.T})

    problem = sim.create_problem()
    assert problem.y == np.array([1])
    assert problem.p == np.array([1])
    assert problem.scale == [1 * u.m * u.s]

    problem = sim.create_problem(values={Model.x: 10 * u.cm, Model.T: 10 * u.ms})
    assert problem.y == np.array([0.1])
    assert problem.p == np.array([0.01])
    assert problem.scale == [100 * u.cm * 1000 * u.ms]


def test_simulator_values():
    class Model(System):
        x: Variable = initial(default=1)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << -x / T

    sim = Simulator(Model)

    with raises(DimensionalityError):
        sim.solve(save_at=range(3), values={Model.x: 1 * u.m})

    with raises(DimensionalityError):
        sim.solve(save_at=range(3), values={Model.T: 1})

    sim.solve(save_at=range(3), values={Model.T: 1 * u.ms})


def test_normalization():
    class Model(System):
        T: Parameter = assign(default=1 * u.s)
        x: Variable = initial(default=1 * u.m)
        y: Variable = initial(default=1 * u.m)
        eq_x = x.derive() << (x - y) / T
        eq_y = y.derive() << (x + y) / T

    t = np.linspace(0, 1, 10)
    sim = Simulator(Model)
    df = sim.solve(save_at=t)
    df_cm = sim.solve(values={Model.y: 100 * u.cm}, save_at=t)
    assert np.allclose((df - df_cm).pint.dequantify().values, 0)
    assert df["y"].pint.units == u.m
    assert df_cm["y"].pint.units == u.cm


@mark.xfail(reason="Not yet implemented")
def test_unit_in_equation():
    class Model(System):
        x: Variable = initial(default=0 * u.m)
        eq = x.derive() << x / (1 * u.s)

    sim = Simulator(Model)
    sim.solve(save_at=np.linspace(0, 1, 10))


def test_zero_initial_with_unit():
    class Model(System):
        x: Variable = initial(default=0 * u.m)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << x / T

    times = np.linspace(0, 1, 10)
    sim = Simulator(Model)
    sim.solve(save_at=times)


def test_v_derive():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=0 * u.m/u.s)
        w: Parameter = assign(default=1 * u.Hz)
        eq = v.derive() << -w**2 * x