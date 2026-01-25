import numpy as np
import pint
from pytest import mark

from ...simulator import Simulator
from ...types import Constant, System, Variable, assign, initial
from ..reactions import MassAction, RateLaw, ReactionVariable, reaction_initial

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
    sim = Simulator(model)
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


def test_reaction_variable():
    class Model(System):
        A: ReactionVariable = reaction_initial(default=1)
        B: ReactionVariable = reaction_initial(default=1)
        AB: ReactionVariable = reaction_initial(default=0)

        eq = MassAction(reactants=[A, 2 * B], products=[AB], rate=A.variable ** (1 / 2))

    assert set(Model._yield(Variable)) == set(
        [Model.A.variable, Model.B.variable, Model.AB.variable]
    )
    dnsim = Simulator(Model)
    assert set(dnsim.compiled.variables) == set(
        [Model.A.variable, Model.B.variable, Model.AB.variable]
    )
    assert np.all(
        dnsim.compiled.func(0, [4, 0, 2], [], [0, 0, 0]) == np.array([-32, 32, -64])
    )
    dnsim.solve(save_at=np.linspace(0, 10, 10))


def test_nested_reaction_variable():
    class DoubleNested(System):
        A: ReactionVariable = reaction_initial(default=1)
        B: ReactionVariable = reaction_initial(default=1)
        AB: ReactionVariable = reaction_initial(default=0)

        eq = MassAction(reactants=[A, 2 * B], products=[AB], rate=A.variable ** (1 / 2))

    class Nested(System):
        A: ReactionVariable = reaction_initial(default=3)
        B: ReactionVariable = reaction_initial(default=0)

        dnested = DoubleNested(A=2)
        eq = RateLaw(reactants=[A], products=[B], rate_law=0.1)

    class Model(System):
        A: ReactionVariable = reaction_initial(default=1)
        B: ReactionVariable = reaction_initial(default=2)

        nested = Nested(A=3 * A)
        eq = MassAction(reactants=[A], products=[B], rate=0.2)

    assert set(Model._yield(Variable)) == set(
        [
            Model.A.variable,
            Model.B.variable,
            Model.nested.B.variable,
            Model.nested.dnested.A.variable,
            Model.nested.dnested.B.variable,
            Model.nested.dnested.AB.variable,
        ]
    )
    assert Model.nested.A.stoichiometry == 3

    sim = Simulator(Model)

    assert set(sim.compiled.variables) == set(
        [
            Model.A.variable,
            Model.B.variable,
            Model.nested.B.variable,
            Model.nested.dnested.A.variable,
            Model.nested.dnested.B.variable,
            Model.nested.dnested.AB.variable,
        ]
    )

    sim.solve(save_at=np.linspace(0, 10, 10))
