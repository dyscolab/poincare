import matplotlib.pyplot as plt
import numpy as np
from symbolite import real

from poincare import (
    Derivative,
    Independent,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
)


class Oscillator(System):
    t = Independent()
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)
    phase: Parameter = assign(default=0)
    F: Parameter = assign(default=real.cos(t + phase))
    spring = vx.derive() << -x + F


if __name__ == "__main__":
    save_at = np.linspace(0, 50, 1000)
    sim = Simulator(Oscillator).with_times(save_at=save_at)
    result = sim.solve()
    result.to_dataframe().plot()

    # No recompilation necessary to change parameter value
    result = sim.with_values({Oscillator.phase: -1.3}).solve()
    result.to_dataframe().plot()

    # To change functional form, must be recompiled
    sim = Simulator(Oscillator(F=real.sin(Oscillator.t))).with_times(save_at=save_at)
    result = sim.solve()
    result.to_dataframe().plot()

    plt.legend()
    plt.show()
