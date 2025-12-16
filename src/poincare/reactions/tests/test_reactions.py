from ..reactions import RateLaw, ReactionVariable, MassAction

from ...types import System, Variable, Constant, Parameter, initial, assign
from ...simulator import Simulator

from pytest import mark
import numpy as np


def non_instance(model):
    return model


def instance(model):
    return model()


@mark.parametrize("f", [non_instance, instance])
def test_reaction(f):
    class Model(System):
        x: Variable = initial(default=0)
        c: Constant = assign(default=0, constant=True)
        eq1 = RateLaw(reactants=[x], products=[], rate_law=c)
        eq2 = RateLaw(reactants=[], products=[x], rate_law=c)
        eq3 = RateLaw(reactants=[2 * x], products=[3 * x], rate_law=c)

    model: Model = f(Model)
    assert model.x.equation_order == 1
    assert set(model.eq1.equations) == {model.x.derive() << -1 * model.c}
    assert set(model.eq2.equations) == {model.x.derive() << 1 * model.c}
    assert set(model.eq3.equations) == {model.x.derive() << 1 * model.c}


@mark.parametrize("f", [non_instance, instance])
def test_reaction_simulation(f):
    class Model(System):
        x: Variable = initial(default=0)
        c: Constant = assign(default=1, constant=True)
        eq1 = RateLaw(reactants=[x], products=[], rate_law=c)
        eq2 = RateLaw(reactants=[], products=[x], rate_law=c)
        eq3 = RateLaw(reactants=[2 * x], products=[3 * x], rate_law=c)

    model: Model = f(Model)
    assert model.x.equation_order == 1
    assert set(model.eq1.equations) == {model.x.derive() << -1 * model.c}
    assert set(model.eq2.equations) == {model.x.derive() << 1 * model.c}
    assert set(model.eq3.equations) == {model.x.derive() << 1 * model.c}

    sim = Simulator(Model)
    result = sim.solve(save_at=np.linspace(0, 1, 5))
    assert result.columns.tolist() == ["x"]
    assert sim.compiled.func(0, [2], [0], [0]) == np.array([1.0])


@mark.parametrize("f", [non_instance, instance])
def test_mass_action(f):
    class Model(System):
        x: Variable = initial(default=0)
        c: Constant = assign(default=0, constant=True)
        eq1 = MassAction(reactants=[x], products=[], rate=c)
        eq2 = MassAction(reactants=[], products=[x], rate=c)
        eq3 = MassAction(reactants=[2 * x], products=[3 * x], rate=c)

    model: Model = f(Model)
    assert model.x.equation_order == 1
    assert set(model.eq1.equations) == {
        model.x.derive() << -1 * (model.c * (model.x**1))
    }
    assert set(model.eq2.equations) == {model.x.derive() << 1 * model.c}
    assert set(model.eq3.equations) == {
        model.x.derive() << 1 * (model.c * (model.x**2))
    }


@mark.parametrize("f", [non_instance, instance])
def test_mass_action_in_simulator(f):
    class Model(System):
        x: Variable = initial(default=0)
        c: Constant = assign(default=2, constant=True)
        eq1 = MassAction(reactants=[x], products=[], rate=c)
        eq2 = MassAction(reactants=[], products=[x], rate=c)
        eq3 = MassAction(reactants=[2 * x], products=[3 * x], rate=c)

    model: Model = f(Model)
    sim = Simulator(Model)
    result = sim.solve(save_at=np.linspace(0, 1, 5))
    assert result.columns.tolist() == ["x"]
    assert sim.compiled.func(0, [2], [0], [0]) == np.array([6.0])  # TODO right result?
