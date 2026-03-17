import numpy as np

from ...simulator import Simulator
from ...types import Parameter, System, Variable, assign, initial
from ..reactions import MassAction
from ..rebop import RebopSimulator


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

    sim = Simulator(Model)
    sol = sim.solve(save_at=np.arange(0, 51, 1))
    rsim = RebopSimulator(Model)
    r_sol = rsim.solve(n_points=50, upto_t=50, rng=1)
    assert ((sol - r_sol) / (sol + r_sol) * 2 <= 0.05).to_array().all()
