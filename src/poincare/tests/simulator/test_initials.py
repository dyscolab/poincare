from pytest import mark, raises
from symbolite import Real

from ...simulator import Simulator
from ...types import (
    Constant,
    Derivative,
    Independent,
    Number,
    Parameter,
    System,
    Variable,
    assign,
    initial,
)


class Model(System):
    k0 = Constant(default=0)
    k1 = Constant(default=k0)
    k2 = Constant(default=k1)
    k3 = Constant(default=k1 + k2)
    k4 = Constant(default=2 * k1)
    p = Parameter(default=k3 + 1)
    x = Variable(initial=k3 + k4)
    eq = x.derive() << p * x


def assert_initials(
    system: System | type[System],
    values: dict[Constant | Parameter | Variable | Derivative, Number | Real],
    *,
    expected_parameters: dict[Parameter, float],
    expected_variables: dict[Variable, float],
):
    sim = Simulator(system)
    problem = sim.with_values(values).create_problem()
    assert dict(zip(sim.compiled.parameters, problem.p)) == expected_parameters
    assert dict(zip(sim.compiled.variables, problem.y)) == expected_variables


def test_default_initials():
    assert_initials(
        Model,
        values={},
        expected_parameters={Model.p: 1},
        expected_variables={Model.x: 0},
    )


def test_override_initials():
    assert_initials(
        Model,
        values={Model.k3: 1},
        expected_parameters={Model.p: 2},
        expected_variables={Model.x: 1},
    )
    assert_initials(
        Model,
        values={Model.k2: 1},
        expected_parameters={Model.p: 2},
        expected_variables={
            Model.x: 1,
        },
    )
    assert_initials(
        Model,
        values={Model.k1: 1},
        expected_parameters={Model.p: 3},
        expected_variables={
            Model.x: 4,
        },
    )
    assert_initials(
        Model,
        values={Model.k0: 1},
        expected_parameters={Model.p: 3},
        expected_variables={
            Model.x: 4,
        },
    )
    assert_initials(
        Model,
        values={Model.k0: 1, Model.k2: 2},
        expected_parameters={Model.p: 4},
        expected_variables={
            Model.x: 5,
        },
    )


def test_override_with_constants():
    assert_initials(
        Model,
        values={Model.k0: 2, Model.k1: 1, Model.k2: Model.k0},
        expected_parameters={Model.p: 4},
        expected_variables={Model.x: 5},
    )


def test_cyclic_initials():
    with raises(ValueError, match="Cyclic"):
        Simulator(Model).with_values({Model.k0: Model.k2}).create_problem()


@mark.xfail(
    reason="Behaviour not expected anymore, variable is know converted to parameter"
)
def test_variable_as_parameter():
    """Variable used in equation but without its own rate equation is a Parameter."""

    class Model(System):
        x = Variable(initial=1)
        T = Variable(initial=2)
        eq = x.derive() << -x / T

    assert_initials(
        Model,
        values={Model.x: 1, Model.T: 2},
        expected_parameters={Model.T: 2},
        expected_variables={Model.x: 1},
    )


def test_time_dependent_parameter_outside_rate_equations():
    class Model(System):
        t = Independent()
        c = Constant(default=0)
        k = Parameter(default=c * t)
        x = Variable(initial=0)
        eq = x.derive() << -x

    Simulator(Model)


def test_none_initial_variable():
    class Model(System):
        x: Variable = initial()
        eq = x.derive() << -x

    sim = Simulator(Model)
    with raises(TypeError, match="Missing initial value"):
        sim.solve(save_at=[0, 1])

    with raises(TypeError, match="Missing initial value"):
        sim.create_problem()

    res = sim.with_values({Model.x: 1.0}).solve(save_at=[0, 1])
    assert res["x"].values[0] == 1.0


def test_none_initial_parameter():
    class Model(System):
        x: Variable = initial(default=1.0)
        p: Parameter = assign()
        eq = x.derive() << -p * x

    sim = Simulator(Model)
    with raises(TypeError, match="Missing initial value"):
        sim.solve(save_at=[0, 1])

    with raises(TypeError, match="Missing initial value"):
        sim.create_problem()

    res = sim.with_values({Model.p: 2.0}).solve(save_at=[0, 1])
    assert res["x"].values[0] == 1.0


def test_none_initial_derivative():
    class Model(System):
        x: Variable = initial(default=1.0)
        v: Derivative = x.derive(initial=None)
        eq = v.derive() << -x

    sim = Simulator(Model)
    with raises(TypeError, match="Missing initial value"):
        sim.solve(save_at=[0, 1])

    res = sim.with_values({Model.x.derive(): 0.0}).solve(save_at=[0, 1])
    assert res["x"].values[0] == 1.0


def test_none_initial_dependent():
    class Model(System):
        c: Constant = assign(default=None, constant=True)
        x: Variable = initial(default=2 * c)
        eq = x.derive() << -x

    sim = Simulator(Model)
    with raises(TypeError, match="Missing initial value"):
        sim.solve(save_at=[0, 1])

    res = sim.with_values({Model.c: 1.0}).solve(save_at=[0, 1])
    assert res["x"].values[0] == 2.0
