from pytest import raises

from ... import Constant, Parameter, System, Variable, assign, initial
from ...simulator import Simulator


def assert_names(
    system: System | type[System],
    *,
    variables: set[str],
    parameters: set[str],
):
    sim = Simulator(system)
    ds = sim.solve(save_at=range(1))
    print(sim.compiled)
    assert set(map(str, sim.compiled.variables)) == variables
    assert set(map(str, sim.compiled.parameters)) == parameters
    assert set(ds.data_vars) == variables


def test_no_equation():
    class Model(System):
        c: Constant = assign(default=0, constant=True)
        p: Parameter = assign(default=0)
        x: Variable = initial(default=0)

    assert_names(Model, variables=set(), parameters=set())


def test_single_variable():
    class Model(System):
        c: Constant = assign(default=0, constant=True)
        p: Parameter = assign(default=0)
        x: Variable = initial(default=0)
        eq = x.derive() << 0

    assert_names(Model, variables={"x"}, parameters=set())


def test_variable_and_parameter():
    class Model(System):
        c: Constant = assign(default=0, constant=True)
        p: Parameter = assign(default=0)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    assert_names(Model, variables={"x"}, parameters={"p"})


def test_derivative():
    class Model(System):
        x: Variable = initial(default=0)
        v = x.derive(initial=0)
        eq = v.derive() << 0

    assert_names(Model, variables={"x", "v"}, parameters=set())


def test_derivative_by_composition():
    class Sub1(System):
        x: Variable = initial()
        x1 = x.derive(initial=0)
        eq = x1.derive() << 0

    class SubD(System):
        x: Variable = initial()
        dx = x.derive(initial=0)
        eq = dx.derive() << 0

    with raises(TypeError, match="derivative"):

        class ImplicitDerivative(System):
            x: Variable = initial(default=0)
            s = Sub1(x=x)

    class ExplicitDerivativeBefore(System):
        x: Variable = initial(default=0)
        v = x.derive(initial=0)
        s = SubD(x=x)

    assert_names(ExplicitDerivativeBefore, variables={"x", "v"}, parameters=set())

    with raises(TypeError, match="derivative"):

        class ExplicitDerivativeAfter(System):
            x: Variable = initial(default=0)
            s = SubD(x=x)
            v = x.derive(initial=0)

    with raises(TypeError, match="derivative"):

        class CollidingImplicitDerivative(System):
            x: Variable = initial(default=0)
            s1 = Sub1(x=x)
            sd = SubD(x=x)
