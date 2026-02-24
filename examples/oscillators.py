import matplotlib.pyplot as plt
import numpy as np

from poincare import Derivative, Parameter, Simulator, System, Variable, assign, initial


class Oscillator(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    spring_constant: Parameter = assign(default=0)

    spring = vx.derive() << -spring_constant * x


class Dampening(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    damp_rate: Parameter = assign(default=0)

    dampening = vx.derive() << -damp_rate * vx


class DampedOscillator(System):
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)

    spring_constant: Parameter = assign(default=1)
    damp_rate: Parameter = assign(default=0.1)

    oscillator = Oscillator(x=x, spring_constant=spring_constant)
    dampening = Dampening(x=x, damp_rate=damp_rate)


if __name__ == "__main__":
    sim = Simulator(DampedOscillator)
    result = sim.solve(save_at=np.linspace(0, 50, 1000))
    result.to_dataframe().plot()
    plt.show()
