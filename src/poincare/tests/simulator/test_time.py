from pytest import raises

from ...simulator import Simulator
from ...types import Constant, Independent, Parameter, System, Variable, assign, initial


def test_time():
    class Model(System):
        t = Independent()
        p: Parameter = assign(default=t)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    problem = Simulator(Model).create_problem()

    with raises(ValueError, match="recompile"):
        Simulator(Model).with_values({Model.p: 2 * Model.t}).create_problem()

    problem_2 = Simulator(Model(p=2 * Model.t)).create_problem()

    assert problem.y == problem_2.y
    assert len(problem.p) == len(problem_2.p) == 0


def test_no_time_dependent_parameters():
    class Model(System):
        t = Independent()
        c0: Constant = assign(default=0, constant=True)
        c1: Constant = assign(default=c0, constant=True)
        p: Parameter = assign(default=c1)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    sim = Simulator(Model)
    sim.solve(save_at=range(2))
    assert set(sim.compiled.parameters) == {Model.p}

    sim.with_values({Model.p: 1}).create_problem()

    with raises(ValueError, match="recompile"):
        sim.with_values({Model.p: Model.t}).create_problem()

    Simulator(Model(p=Model.t))


def test_time_dependent_parameters():
    class Model(System):
        t = Independent()
        p: Parameter = assign(default=t)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    sim = Simulator(Model)
    sim.solve(save_at=range(2))
    assert len(sim.compiled.parameters) == 0

    with raises(ValueError, match="recompile"):
        sim.with_values({Model.p: 1}).create_problem()

    with raises(ValueError, match="recompile"):
        sim.with_values({Model.p: Model.t}).create_problem()

    Simulator(Model(p=Model.t))


def test_variable_dependent_parameters():
    class Model(System):
        x: Variable = initial(default=0)
        eq = x.derive() << 1  # analogous to "time"
        p: Parameter = assign(default=x)
        y: Variable = initial(default=0)
        eq2 = y.derive() << p

    sim = Simulator(Model)
    sim.solve(save_at=range(2))
    assert len(sim.compiled.parameters) == 0

    with raises(ValueError, match="recompile"):
        sim.with_values({Model.p: 1}).create_problem()


def test_parameter_dependent_parameters():
    class Model(System):
        t = Independent()
        p0: Parameter = assign(default=0)
        p: Parameter = assign(default=p0)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    sim = Simulator(Model)
    sim.solve(save_at=range(2))
    # only p is part of the vector of parameters
    assert set(sim.compiled.parameters) == {Model.p}
    # but initial values can be modified through p0
    assert sim.create_problem().p[0] == 0
    assert sim.with_values({Model.p: 1}).create_problem().p[0] == 1
    assert sim.with_values({Model.p0: 1}).create_problem().p[0] == 1

    func = Model.t

    # must recompile to assign a function to p or p0
    with raises(ValueError, match="recompile"):
        sim.with_values({Model.p: func}).create_problem()

    with raises(ValueError, match="recompile"):
        sim.with_values({Model.p0: func}).create_problem()

    # recompilation moves from parameter vector to parameter func
    model = Model(p=func)
    sim = Simulator(model)
    assert len(sim.compiled.parameters) == 0

    # recompilation moves from parameter vector to parameter func
    model = Model(p=func * Model.p0)
    sim = Simulator(model)
    assert len(sim.compiled.parameters) == 1

    # recompilation moves from parameter vector to parameter func
    model = Model(p0=func)
    sim = Simulator(model)
    assert len(sim.compiled.parameters) == 0


def test_parameter_dependent_parameters2():
    class Model(System):
        t = Independent()
        p0: Parameter = assign(default=0)
        p1: Parameter = assign(default=p0)
        p: Parameter = assign(default=p0 * p1)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    sim = Simulator(Model)
    sim.solve(save_at=range(2))
    assert set(sim.compiled.parameters) == {Model.p}

    model = Model(p0=Model.t)
    sim = Simulator(model)
    sim.solve(save_at=range(2))
    assert len(sim.compiled.parameters) == 0
