"""Harmonic oscillator in poincare, with units, checked against x(t) = x0 cos(w t).

Shows: Variable/Derivative/Parameter declaration, a second-order equation,
pint units, tight solver tolerances, simulation times, transforms and
dequantified output.
"""

import numpy as np
import pint

from poincare import Derivative, Parameter, Simulator, System, Variable, assign, initial
from poincare import solvers

u = pint.get_application_registry()


class Oscillator(System):
    x: Variable = initial(default=1 * u.m)
    v: Derivative = x.derive(initial=0 * u.m / u.s)
    w: Parameter = assign(default=2 / u.s)

    spring = v.derive() << -(w**2) * x


save_at = np.linspace(0, 10, 201) * u.s
sim = (
    Simulator(Oscillator)
    .with_solver(solvers.LSODA(rtol=1e-9, atol=1e-12))
    .with_times(save_at=save_at)
    .with_transform(
        {"energy": 0.5 * Oscillator.v**2 + 0.5 * Oscillator.w**2 * Oscillator.x**2},
        append=True,  # keep x and v in the output as well
    )
)
result = sim.solve()

# Physics checks: analytic solution and energy conservation.
t = save_at.to(u.s).magnitude
x = result["x"].pint.to(u.m).pint.dequantify().values
np.testing.assert_allclose(x, np.cos(2 * t), atol=1e-6)
energy = result["energy"].pint.dequantify().values
np.testing.assert_allclose(energy, energy[0], rtol=1e-6)

# A different parameter value reuses the compiled model (and the times).
fast = sim.with_values({Oscillator.w: 4 / u.s}).solve()
np.testing.assert_allclose(
    fast["x"].pint.dequantify().values, np.cos(4 * t), atol=1e-6
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result.pint.dequantify().to_dataframe()[["x", "v"]].plot()
    plt.show()
