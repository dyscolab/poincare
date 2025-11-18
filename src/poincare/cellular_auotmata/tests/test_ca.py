from ..ca_simulator import CASimulator
from ..ca_types import Automata
from ...types import System, Variable, Derivative, Parameter, initial, assign


class Oscillator(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    spring_constant: Parameter = assign(default=0)

    spring = vx.derive() << -spring_constant * x


class Coupling(System):
    # Create the variables for both oscillators
    x_1: Variable = initial(default=0)
    v_1: Derivative = x_1.derive(initial=0)
    x_2: Variable = initial(default=0)
    v_2: Derivative = x_2.derive(initial=0)

    spring_constant: Parameter = assign(default=0.1)

    # Apply the force from the interaction to both springs
    force_1 = v_1.derive() << spring_constant * (x_2 - x_1)
    force_2 = v_2.derive() << spring_constant * (x_1 - x_2)


def test_initials():
    OscilatorChain = Automata(cell=Oscillator, interact=Coupling, boundary=None)
    ca_sim = CASimulator(OscilatorChain, shape=(5,))
    assert True
