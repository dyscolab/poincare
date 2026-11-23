"""Steady states and parameter sweeps in poincare, checked against x* = p / k.

Shows: SteadyState().solve, SteadyState().sweep (unitless state variables),
and the loop + xr.concat pattern, which also works when variables carry
units or the simulator has transforms.
"""

import numpy as np
import pint
import xarray as xr

from poincare import Parameter, Simulator, SteadyState, System, Variable, assign, initial
from poincare import solvers

u = pint.get_application_registry()

# SteadyState stops when derivatives are small relative to the solver
# tolerances, so tight tolerances give accurate steady states.
TIGHT = solvers.LSODA(rtol=1e-9, atol=1e-12)


class Production(System):
    x: Variable = initial(default=0)
    p: Parameter = assign(default=1.0)
    k: Parameter = assign(default=2.0)

    production = x.derive() << p
    degradation = x.derive() << -k * x  # equations for the same derivative add up


sim = Simulator(Production).with_solver(TIGHT)

single = SteadyState().solve(sim)
assert single["event"].item() == 0  # 0: converged before t_end
np.testing.assert_allclose(single["x"].item(), 0.5, rtol=1e-6)

# Built-in sweep: fine for state variables without units.
ps = np.linspace(0.5, 3, 6)
swept = SteadyState().sweep(sim, variable=Production.p, values=ps)
np.testing.assert_allclose(swept["x"].values, ps / 2, rtol=1e-6)


# Loop pattern: works with units and with transforms.
class ProductionWithUnits(System):
    x: Variable = initial(default=0 * u.mM)
    p: Parameter = assign(default=1 * u.mM / u.s)
    k: Parameter = assign(default=2 / u.s)

    production = x.derive() << p - k * x


sim_u = (
    Simulator(ProductionWithUnits)
    .with_solver(TIGHT)
    .with_transform({"flux_out": ProductionWithUnits.k * ProductionWithUnits.x}, append=True)
)
p_values = [0.5, 1, 2] * u.mM / u.s
sweep = xr.concat(
    # isel(time=-1): each run stops at a different time; without it concat
    # aligns on time and fills the gaps with NaN.
    [
        SteadyState().solve(sim_u.with_values({ProductionWithUnits.p: p})).isel(time=-1)
        for p in p_values
    ],
    dim="p",
).assign_coords(p=("p", p_values.magnitude, {"units": str(p_values.units)}))

expected = (p_values / (2 / u.s)).to(u.mM).magnitude
np.testing.assert_allclose(
    sweep["x"].pint.to(u.mM).pint.dequantify().values.ravel(), expected, rtol=1e-6
)
# At steady state, outflow balances production.
np.testing.assert_allclose(
    sweep["flux_out"].pint.to(u.mM / u.s).pint.dequantify().values.ravel(),
    p_values.magnitude,
    rtol=1e-6,
)
