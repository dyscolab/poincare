"""Regression: scipy must not receive the shared derivative buffer.

scipy keeps the arrays the right-hand side returns (for example f(t, y) while
it estimates a Jacobian by finite differences). Returning one shared buffer
let later calls overwrite them, so BDF and Radau failed on stiff systems.
"""

from dataclasses import replace

import numpy as np
from pytest import mark

from ... import Parameter, System, Variable, solvers
from ...simulator import Simulator


class Robertson(System):
    """Stiff chemical kinetics, with a conserved total."""

    a = Variable(initial=1)
    b = Variable(initial=0)
    c = Variable(initial=0)
    k1 = Parameter(default=0.04)
    k2 = Parameter(default=3e7)
    k3 = Parameter(default=1e4)
    eq_a = a.derive() << -k1 * a + k3 * b * c
    eq_b = b.derive() << k1 * a - k3 * b * c - k2 * b**2
    eq_c = c.derive() << k2 * b**2


@mark.parametrize(
    "solver",
    [solvers.BDF(), solvers.Radau(), solvers.RK45()],
    ids=lambda s: type(s).__name__,
)
def test_scipy_receives_a_new_derivative_array(solver):
    if isinstance(solver, solvers.RK45):
        t = np.linspace(0, 1, 5)
    else:
        t = np.logspace(-5, 5, 20)
    result = (
        Simulator(Robertson)
        .with_solver(replace(solver, rtol=1e-6, atol=1e-10))
        .solve(save_at=t)
    )
    total = result["a"] + result["b"] + result["c"]
    np.testing.assert_allclose(total, 1, rtol=1e-5)
