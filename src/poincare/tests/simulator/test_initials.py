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
    problem = sim.create_problem(values)
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
        Simulator(Model).create_problem(values={Model.k0: Model.k2})


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
