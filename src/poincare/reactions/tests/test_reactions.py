from ..reactions import RateLaw, ReactionVariable, MassAction

from ...types import System, Variable, Constant, Parameter, initial, assign
from ...simulator import Simulator

from pytest import mark
import numpy as np
import pint

ureg = pint.UnitRegistry()


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
    assert sim.compiled.func(0, [2], [0], [0]) == np.array([6.0])


def test_units_in_rate_law():
    class Model(System):
        x: Variable = initial(default=0)
        c: Constant = assign(default=0, constant=True)
        eq1 = RateLaw(reactants=[x], products=[], rate_law=c)
        eq2 = RateLaw(reactants=[], products=[x], rate_law=c)
        eq3 = RateLaw(reactants=[2 * x], products=[3 * x], rate_law=c)

    class UnitModel(System):
        x: Variable = initial(default=0 * ureg.mol)

        eq1 = RateLaw(reactants=[x], products=[], rate_law=1 * ureg.mol / ureg.s)
        eq2 = RateLaw(reactants=[], products=[x], rate_law=1 * ureg.mol / ureg.s)
        eq3 = RateLaw(
            reactants=[2 * x], products=[3 * x], rate_law=1 * ureg.mol / ureg.s
        )

    assert UnitModel._eq1_rate_law.default == 1 * ureg.mol / ureg.s
    sim_1 = Simulator(Model)
    result_1 = np.asarray(sim_1.solve(save_at=np.linspace(0, 10, 10)))
    sim_2 = Simulator(Model)
    result_2 = np.asarray(sim_2.solve(save_at=np.linspace(0, 10, 10)).pint.dequantify())
    assert np.all(result_1 == result_2)


def test_units_in_mass_action():
    class Model(System):
        x: Variable = initial(default=0)
        c: Constant = assign(default=0, constant=True)
        eq1 = MassAction(reactants=[x], products=[], rate=c)
        eq2 = MassAction(reactants=[], products=[x], rate=c)
        eq3 = MassAction(reactants=[2 * x], products=[3 * x], rate=c)

    class UnitModel(System):
        x: Variable = initial(default=0 * ureg.mol)

        eq1 = MassAction(reactants=[x], products=[], rate=1 / ureg.s)
        eq2 = MassAction(reactants=[], products=[x], rate=1 * ureg.mol / ureg.s)
        eq3 = MassAction(
            reactants=[2 * x], products=[3 * x], rate=1 / ureg.s / ureg.mol
        )

    assert UnitModel._eq2_rate_law.default == 1 * ureg.mol / ureg.s
    sim_1 = Simulator(Model)
    result_1 = np.asarray(sim_1.solve(save_at=np.linspace(0, 10, 10)))
    sim_2 = Simulator(Model)
    result_2 = np.asarray(sim_2.solve(save_at=np.linspace(0, 10, 10)).pint.dequantify())
    assert np.all(result_1 == result_2)
