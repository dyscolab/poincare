import numpy as np
from symbolite import real

from ...simulator import Simulator
from ...types import Constant, Parameter, System, Variable, assign, initial
from ..reactions import MassAction, RateLaw
from ..rebop.rebop import RebopSimulator


def compare_rebop_and_ode(model: type[System], values={}):
    sim = Simulator(model)
    sol = sim.solve(save_at=np.arange(0, 51, 1), values=values)
    rsim = RebopSimulator(model)
    r_sol = rsim.solve(n_points=50, upto_t=50, rng=1, values=values)
    tolerance = 0.1
    assert np.abs(((sol - r_sol) / (sol + r_sol) * 2 <= tolerance).to_array()).all()


def test_rebop():
    class Model(System):
        A: Variable = initial(default=1e4)
        B: Variable = initial(default=1e4)
        AB: Variable = initial(default=100)
        reaction_rate: Parameter = assign(default=1e-9)

        eq1 = MassAction(
            reactants=[A, 2 * B],
            products=[AB],
            rate=reaction_rate,
        )
        eq2 = MassAction(reactants=[], products=[B], rate=1e-9)
        eq3 = MassAction(reactants=[B], products=[], rate=1e-9)

    compare_rebop_and_ode(Model)


def test_arbitrary_rate_in_rebop():
    class Model(System):
        A: Variable = initial(default=1e4)
        B: Variable = initial(default=1e4)
        AB: Variable = initial(default=100)
        reaction_rate: Parameter = assign(
            default=(A**2 + B) / (B + 1) + real.sqrt(B) ** 2
        )

        eq1 = MassAction(
            reactants=[A, 2 * B],
            products=[AB],
            rate=1e-14 * reaction_rate,
        )
        eq2 = RateLaw(reactants=[], products=[B], rate_law=1e-14 * reaction_rate)

    compare_rebop_and_ode(Model)


def test_no_non_reaction_equations():
    class Model(System):
        A: Variable = initial(default=1e4)
        B: Variable = initial(default=1e4)
        eq1 = MassAction(
            reactants=[A],
            products=[B],
            rate=1,
        )
        eq2 = A.derive() << 1

    try:
        RebopSimulator(Model)
    except TypeError:
        return
    else:
        assert False


def test_changed_initials():
    class Model(System):
        c: Constant = assign(default=1, constant=True)
        A_0: Constant = assign(default=1e4, constant=True)
        A: Variable = initial(default=A_0)
        B: Variable = initial(default=1000)
        p: Parameter = assign(default=A**2 * c)

        r1 = MassAction(reactants=[A], products=[B], rate=1e-6 * c)
        r2 = RateLaw(reactants=[A], products=[B], rate_law=1e-6 * c * p)

    compare_rebop_and_ode(Model)
    compare_rebop_and_ode(Model, values={Model.A_0: 2e4, Model.B: 2000, Model.c: 2})
